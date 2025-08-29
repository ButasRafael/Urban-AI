from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
from openai import AsyncOpenAI
import json
import os
import re

from app.models.media import IssueStatus, Severity
from app.models.user import User
from app.core.database import SessionLocal
from app.core.datetime_utils import RO_TZ

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_authorities_from_db() -> List[str]:

    db = SessionLocal()
    try:
        authorities = db.query(User.username).filter(User.role == "authority").all()
        return [auth[0] for auth in authorities] if authorities else []
    finally:
        db.close()

def get_admins_from_db() -> List[str]:

    db = SessionLocal()
    try:
        admins = db.query(User.username).filter(User.role == "admin").all()
        return [admin[0] for admin in admins] if admins else []
    finally:
        db.close()

async def parse_query_with_filters(user_query: str, current_time: Optional[datetime] = None) -> Dict[str, Any]:

    # Get list of valid authorities and admins from database
    authorities = get_authorities_from_db()
    authorities_str = "\n".join(f"  • {auth}" for auth in authorities) if authorities else "  • No authorities available"
    
    admins = get_admins_from_db()
    admins_str = "\n".join(f"  • {admin}" for admin in admins) if admins else "  • No admins available"

    # Get current time for temporal context - use Romanian timezone
    if not current_time:
        current_time = datetime.now(RO_TZ)
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M %Z")
    current_date_str = current_time.strftime("%B %d, %Y")

    # Calculate some useful reference dates
    yesterday = current_time - timedelta(days=1)
    last_week = current_time - timedelta(days=7)
    last_month = current_time - timedelta(days=30)

    instructions = f"""# Role & Objective
You are a query parser for an urban issue tracking RAG system. Extract:
- A semantic search "query" string that preserves locations, areas, issue types, and descriptive terms
- ONLY the dynamic filters: "severity", "status", "assigned_to", "verified_by", "resolved_after", "resolved_before", "verified_after", "verified_before"
- A boolean flag "sql_only" indicating if the query can be fully resolved using SQL filters alone

# Current Time Context
- Current time: {current_time_str}
- Today's date: {current_date_str}
- Yesterday: {yesterday.strftime('%Y-%m-%d')}
- Last week starts: {last_week.strftime('%Y-%m-%d')}
- Last month starts: {last_month.strftime('%Y-%m-%d')}
- Current year: {current_time.year}

# Instructions
- Retrieval is two-phase:
  1) SQL filters on dynamic fields (severity, status, assigned_to, verified_by, resolved_at, verified_at)
  2) Vector similarity for everything else
- Dynamic filters allowed (ONLY these go in filters):
  • severity: "low" | "medium" | "high"  (lowercase ONLY)
  • status:   "open" | "resolved" | "ignored"  (lowercase ONLY)
  • assigned_to: MUST be one of the valid authorities listed below or null
  • verified_by: MUST be one of the valid admins listed below or null
  • resolved_after: ISO date (YYYY-MM-DD) - issues resolved on or after this date
  • resolved_before: ISO date (YYYY-MM-DD) - issues resolved on or before this date  
  • verified_after: ISO date (YYYY-MM-DD) - issues verified on or after this date
  • verified_before: ISO date (YYYY-MM-DD) - issues verified on or before this date

# Valid Authorities (assigned_to must match one of these EXACTLY):
{authorities_str}

# Valid Admins (verified_by must match one of these EXACTLY):
{admins_str}

# SQL-Only Detection Rules
Set "sql_only": true when ALL of these conditions are met:
1. At least one SQL filter is extracted (severity, status, assigned_to, verified_by, or date filters)
   - IMPORTANT: If ALL filters are null, set sql_only: false regardless of query content
2. The remaining query (after filter extraction) is either:
   - Empty or very short (< 15 characters)
   - Contains only generic terms: "issues", "problems", "probleme", "defects", "items", "reports", "all"
   - Has no location names, street names, areas, neighborhoods, or descriptive content

Examples of SQL-only queries (MUST have at least one filter):
- "high severity issues" → sql_only: true (has severity filter + generic term)
- "resolved problems" → sql_only: true (has status filter + generic term)
- "issues assigned to police" → sql_only: true (has assigned_to filter + generic term)
- "all open items" → sql_only: true (has status filter + generic term)

Examples of queries needing semantic search:
- "issues" → sql_only: false (NO filters extracted, needs semantic search)
- "all problems" → sql_only: false (NO filters, needs semantic search)
- "potholes on Main Street" → sql_only: false (location info needs semantic search)
- "lighting issues in downtown" → sql_only: false (area + issue type need semantic)
- "high severity potholes" → sql_only: false ("potholes" is specific issue type)
- "water leaks near the park" → sql_only: false (issue type + location)

- For assigned_to: If the user mentions an authority name, find the CLOSEST matching authority from the list above and use that EXACT name. If no reasonable match exists, set to null.
- For verified_by: If the user mentions an admin username, find the EXACT matching admin from the list above. If the username doesn't match any admin in the list, set to null
- For temporal filters, interpret natural language using current time context:
  • "resolved recently" / "resolved in the last week" → resolved_after: "{last_week.strftime('%Y-%m-%d')}"
  • "resolved today" → resolved_after: "{current_time.strftime('%Y-%m-%d')}", resolved_before: "{current_time.strftime('%Y-%m-%d')}"
  • "resolved yesterday" → resolved_after: "{yesterday.strftime('%Y-%m-%d')}", resolved_before: "{yesterday.strftime('%Y-%m-%d')}"
  • "resolved last month" → resolved_after: "{last_month.strftime('%Y-%m-%d')}"
  • "resolved on January 21" → resolved_after: "2025-01-21", resolved_before: "2025-01-21" (use current year if not specified)
  • "verified this week" → verified_after: "{last_week.strftime('%Y-%m-%d')}"
  • Apply same logic for verified_after/verified_before
  • DO NOT keep time references in the query - they are handled by date filters
- IMPORTANT: Be very strict with filter extraction - ONLY extract filters when explicitly mentioned:
  • Do NOT infer filters from context or pronouns ("assigned to me", "the crew") → keep as null
  • Do NOT assume default values for any filter
  • Extract temporal information ONLY when specifically mentioned
- Everything else MUST remain in "query": streets/addresses/landmarks, areas/neighborhoods/zones, issue types, descriptive terms.
- Normalize synonyms when present (English and Romanian):
  • severity: "critical"/"urgent"/"grave"/"urgente"/"majore"/"severe" → "high"; 
              "moderate"/"medii" → "medium";
              "minor"/"minore"/"usoare" → "low"
  • status: "unresolved"/"pending"/"active"/"nerezolvate"/"deschise" → "open"; 
            "closed"/"fixed"/"rezolvate"/"inchise" → "resolved";
            "ignored"/"ignorate" → "ignored"
- If a filter is not specified, set it to null. Do not invent values.
- Output MUST be a single JSON object with exactly these keys and no others:
  {{"query": "...", "severity": null|"low"|"medium"|"high", "status": null|"open"|"resolved"|"ignored", 
   "assigned_to": null|string, "verified_by": null|string, 
   "resolved_after": null|"YYYY-MM-DD", "resolved_before": null|"YYYY-MM-DD",
   "verified_after": null|"YYYY-MM-DD", "verified_before": null|"YYYY-MM-DD",
   "sql_only": true|false}}

# Examples (assuming "Poliția Locală Cluj-Napoca" and "Primăria Cluj-Napoca" are in the authorities list, "admin_john" is in the admins list)
Input: "high severity potholes in downtown area"
Output: {{"query": "potholes in downtown area", "severity": "high", "status": null, "assigned_to": null, "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": false}}

Input: "high severity issues"
Output: {{"query": "issues", "severity": "high", "status": null, "assigned_to": null, "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": true}}

Input: "probleme grave care nu au fost rezolvate"
Output: {{"query": "probleme", "severity": "high", "status": "open", "assigned_to": null, "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": true}}

Input: "gropi majore pe strada principală nerezolvate de către poliția locală"
Output: {{"query": "gropi pe strada principală", "severity": "high", "status": "open", "assigned_to": "Poliția Locală Cluj-Napoca", "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": false}}

Input: "probleme minore rezolvate săptămâna trecută"
Output: {{"query": "probleme", "severity": "low", "status": "resolved", "assigned_to": null, "verified_by": null, "resolved_after": "{last_week.strftime('%Y-%m-%d')}", "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": true}}

Input: "issues resolved recently by the police"
Output: {{"query": "issues", "severity": null, "status": null, "assigned_to": "Poliția Locală Cluj-Napoca", "verified_by": null, "resolved_after": "{last_week.strftime('%Y-%m-%d')}", "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": true}}

Input: "potholes resolved yesterday verified by admin_john"
Output: {{"query": "potholes", "severity": null, "status": null, "assigned_to": null, "verified_by": "admin_john", "resolved_after": "{yesterday.strftime('%Y-%m-%d')}", "resolved_before": "{yesterday.strftime('%Y-%m-%d')}", "verified_after": null, "verified_before": null, "sql_only": false}}

Input: "iluminat public defect urgente în centru asignat la primărie"
Output: {{"query": "iluminat public defect în centru", "severity": "high", "status": null, "assigned_to": "Primăria Cluj-Napoca", "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": false}}

Input: "scurgeri de apă grave ce țin de primărie"
Output: {{"query": "scurgeri de apă", "severity": "high", "status": null, "assigned_to": "Primăria Cluj-Napoca", "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": false}}

Input: "all open items"
Output: {{"query": "items", "severity": null, "status": "open", "assigned_to": null, "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": true}}

Input: "problems resolved on January 21"
Output: {{"query": "problems", "severity": null, "status": null, "assigned_to": null, "verified_by": null, "resolved_after": "{current_time.year}-01-21", "resolved_before": "{current_time.year}-01-21", "verified_after": null, "verified_before": null, "sql_only": true}}

# Final Instruction
Return ONLY the JSON object for the given input (no backticks, no prose). If you cannot comply, return:
{{"query": "<the original user text>", "severity": null, "status": null, "assigned_to": null, "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null}}
    """.strip()

    try:
        resp = await client.responses.create(
            model="gpt-4.1",
            instructions=instructions,
            input=user_query,
            temperature=0.2,
            max_output_tokens=180,
        )

        raw = (getattr(resp, "output_text", None) or "").strip()

        if not raw:
            raise ValueError("Empty model output")

        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in: {raw!r}")
        obj_text = raw[start:end + 1]

        result = json.loads(obj_text)

        # ---- Server-side normalization / mapping (defensive) ----
        parsed: Dict[str, Any] = {
            "query": result.get("query") or user_query,
            "severity": None,
            "status": None,
            "assigned_to": None,
            "verified_by": None,
            "resolved_after": None,
            "resolved_before": None,
            "verified_after": None,
            "verified_before": None,
            "sql_only": result.get("sql_only", False),
        }

        # Severity
        sev_raw = result.get("severity")
        if isinstance(sev_raw, str):
            sev = sev_raw.lower().strip()
            severity_map = {
                "low": Severity.low,
                "medium": Severity.medium,
                "high": Severity.high,
                # English synonyms
                "critical": Severity.high,
                "urgent": Severity.high,
                "severe": Severity.high,
                "minor": Severity.low,
                "moderate": Severity.medium,
                # Romanian synonyms
                "grave": Severity.high,
                "urgente": Severity.high,
                "majore": Severity.high,
                "severe": Severity.high,
                "medii": Severity.medium,
                "moderate": Severity.medium,
                "minore": Severity.low,
                "usoare": Severity.low,
            }
            parsed["severity"] = severity_map.get(sev)

        # Status
        stat_raw = result.get("status")
        if isinstance(stat_raw, str):
            stat = stat_raw.lower().strip()
            status_map = {
                "open": IssueStatus.open,
                "resolved": IssueStatus.resolved,
                "ignored": IssueStatus.ignored,
                # English synonyms
                "closed": IssueStatus.resolved,
                "fixed": IssueStatus.resolved,
                "unresolved": IssueStatus.open,
                "pending": IssueStatus.open,
                "active": IssueStatus.open,
                # Romanian synonyms
                "nerezolvate": IssueStatus.open,
                "nerezolvat": IssueStatus.open,
                "deschise": IssueStatus.open,
                "deschis": IssueStatus.open,
                "activ": IssueStatus.open,
                "rezolvate": IssueStatus.resolved,
                "rezolvat": IssueStatus.resolved,
                "inchise": IssueStatus.resolved,
                "inchis": IssueStatus.resolved,
                "închise": IssueStatus.resolved,
                "închis": IssueStatus.resolved,
                "ignorate": IssueStatus.ignored,
                "ignorat": IssueStatus.ignored,
            }
            parsed["status"] = status_map.get(stat)

        # Assigned_to - validate it's in the authorities list
        at = result.get("assigned_to")
        if isinstance(at, str):
            name = at.strip()
            # Validate that the assigned_to value is in our authorities list
            if name and name in authorities:
                parsed["assigned_to"] = name
            else:
                # GPT returned an invalid authority, set to null
                parsed["assigned_to"] = None
        else:
            parsed["assigned_to"] = None

        # Verified_by - validate it's in the admins list
        vb = result.get("verified_by")
        if isinstance(vb, str):
            username = vb.strip()
            # Validate that the verified_by value is in our admins list
            if username and username in admins:
                parsed["verified_by"] = username
            else:
                # GPT returned an invalid admin, set to null
                parsed["verified_by"] = None
        else:
            parsed["verified_by"] = None

        # Temporal filters - parse ISO dates
        for field in ["resolved_after", "resolved_before", "verified_after", "verified_before"]:
            date_val = result.get(field)
            if isinstance(date_val, str) and date_val.strip():
                try:
                    # Validate it's a valid date format
                    datetime.strptime(date_val.strip(), "%Y-%m-%d")
                    parsed[field] = date_val.strip()
                except ValueError:
                    # Invalid date format, skip
                    pass

        # Final safety check: If all filters are null, force sql_only to False
        has_any_filter = (
            parsed["severity"] is not None or
            parsed["status"] is not None or
            parsed["assigned_to"] is not None or
            parsed["verified_by"] is not None or
            parsed["resolved_after"] is not None or
            parsed["resolved_before"] is not None or
            parsed["verified_after"] is not None or
            parsed["verified_before"] is not None
        )
        
        if not has_any_filter:
            parsed["sql_only"] = False

        return parsed

    except Exception as e:
        # If parsing fails, treat entire query as semantic search (no filters)
        print(f"Query parsing error: {e}")
        return {
            "query": user_query,
            "severity": None,
            "status": None,
            "assigned_to": None,
            "verified_by": None,
            "resolved_after": None,
            "resolved_before": None,
            "verified_after": None,
            "verified_before": None,
            "sql_only": False,
        }

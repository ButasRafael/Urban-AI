from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, case, desc
from datetime import date, timedelta, datetime, timezone
from typing import Optional, Literal
from app.core.database import get_db
from app.core.security import require_roles
from app.models import media as dbm

try:
    import geohash2 as _geohash_mod
except Exception:
    _geohash_mod = None

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
    dependencies=[require_roles("admin")]
)

# ---- helpers ----
def _start_date(days: int | None, start: Optional[date], end: Optional[date]) -> tuple[datetime, datetime]:
    if start and end:
        start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt   = datetime.combine(end,   datetime.max.time()).replace(tzinfo=timezone.utc)
    else:
        days = 7 if days is None else max(1, days)
        today_utc = datetime.now(timezone.utc).date()
        end_dt   = datetime.combine(today_utc, datetime.max.time()).replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=days-1)
    return start_dt, end_dt


def _media_base(db: Session, start_dt: datetime, end_dt: datetime, media_type: Optional[Literal["image","video"]]):
    q = db.query(dbm.Media).filter(dbm.Media.created_at >= start_dt, dbm.Media.created_at <= end_dt)
    if media_type:
        q = q.filter(dbm.Media.media_type == media_type)
    return q


def _detect_base(db: Session, start_dt: datetime, end_dt: datetime, media_type: Optional[Literal["image","video"]]):
    q = (
        db.query(dbm.Detection, dbm.Media)
          .join(dbm.Frame, dbm.Detection.frame_id == dbm.Frame.id)
          .join(dbm.Media, dbm.Frame.media_id == dbm.Media.id)
          .filter(dbm.Media.created_at >= start_dt, dbm.Media.created_at <= end_dt)
    )
    if media_type:
        q = q.filter(dbm.Media.media_type == media_type)
    return q


# --------------------- uploads ---------------------

@router.get("/uploads-by-day")
def uploads_by_day(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)
    dcol = func.date(dbm.Media.created_at)
    q = (
        _media_base(db, start_dt, end_dt, media_type)
        .with_entities(dcol.label("d"), func.count().label("c"))
        .group_by(dcol)
        .order_by(dcol)
    )
    rows = q.all()
    return [{"date": str(d), "count": c} for d, c in rows]


@router.get("/uploads-by-user")
def uploads_by_user(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)
    q = (
        _media_base(db, start_dt, end_dt, media_type)
        .with_entities(dbm.Media.user_username, func.count())
        .group_by(dbm.Media.user_username)
        .order_by(desc(func.count()))
    )
    rows = q.all()
    return [{"user": u, "count": c} for u, c in rows]


# --------------------- KPIs / latency ---------------------

@router.get("/kpis")
def kpis(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)
    mb = _media_base(db, start_dt, end_dt, media_type)

    total_uploads = mb.count()
    total_images  = mb.filter(dbm.Media.media_type == "image").count()
    total_videos  = mb.filter(dbm.Media.media_type == "video").count()

    lat_q = (
        _media_base(db, start_dt, end_dt, media_type)
        .with_entities(
            func.avg(dbm.Media.process_ms_total).label("avg_ms"),
            func.percentile_cont(0.95).within_group(dbm.Media.process_ms_total).label("p95_ms"),
        )
    ).first()

    detect_q = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(func.count(dbm.Detection.id))
    ).scalar()

    return {
        "window": {"start": start_dt.isoformat(), "end": end_dt.isoformat()},
        "uploads": {"total": total_uploads, "images": total_images, "videos": total_videos},
        "latency_ms": {"avg": float(lat_q.avg_ms or 0), "p95": float(lat_q.p95_ms or 0)},
        "detections": {"total": int(detect_q or 0)},
    }


@router.get("/latency-by-day")
def latency_by_day(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)
    dcol = func.date(dbm.Media.created_at)
    q = (
        _media_base(db, start_dt, end_dt, media_type)
        .with_entities(
            dcol.label("d"),
            func.avg(dbm.Media.process_ms_total).label("avg_ms"),
            func.percentile_cont(0.95).within_group(dbm.Media.process_ms_total).label("p95_ms"),
            func.count().label("n"),
        )
        .group_by(dcol)
        .order_by(dcol)
    )
    rows = q.all()
    return [
        {"date": str(d), "avg_ms": float(avg or 0), "p95_ms": float(p95 or 0), "count": int(n)}
        for d, avg, p95, n in rows
    ]


# --------------------- detections breakdowns ---------------------

@router.get("/detections/severity-by-day")
def detections_severity_by_day(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)
    dcol = func.date(dbm.Media.created_at)
    q = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(dcol.label("d"), dbm.Detection.severity, func.count().label("c"))
        .group_by(dcol, dbm.Detection.severity)
        .order_by(dcol)
    )
    rows = q.all()

    out = {}
    for d, sev, c in rows:
        key = str(d)
        out.setdefault(key, {"date": key, "low": 0, "medium": 0, "high": 0})
        out[key][sev.value if hasattr(sev, "value") else str(sev)] = int(c)

    return [out[k] for k in sorted(out.keys())]


@router.get("/detections/source-breakdown")
def detections_source_breakdown(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)
    q = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(dbm.Detection.source, func.count())
        .group_by(dbm.Detection.source)
    )
    rows = q.all()
    return [{"source": (s.value if hasattr(s, "value") else str(s)), "count": int(c)} for s, c in rows]


@router.get("/detections/status-breakdown")
def detections_status_breakdown(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)
    q = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(dbm.Detection.status, func.count())
        .group_by(dbm.Detection.status)
    )
    rows = q.all()
    return [{"status": (s.value if hasattr(s, "value") else str(s)), "count": int(c)} for s, c in rows]


@router.get("/detections/top-classes")
def detections_top_classes(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
    limit: int = Query(10, ge=1, le=50),
):
    start_dt, end_dt = _start_date(days, start, end)
    q = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(dbm.Detection.class_name, func.count().label("c"))
        .group_by(dbm.Detection.class_name)
        .order_by(desc("c"))
        .limit(limit)
    )
    rows = q.all()
    return [{"class_name": n, "count": int(c)} for n, c in rows]


@router.get("/detections/confidence-summary")
def detections_confidence_summary(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)
    q = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            func.avg(dbm.Detection.confidence).label("avg"),
            func.percentile_cont(0.05).within_group(dbm.Detection.confidence).label("p05"),
            func.percentile_cont(0.50).within_group(dbm.Detection.confidence).label("p50"),
            func.percentile_cont(0.95).within_group(dbm.Detection.confidence).label("p95"),
        )
    ).first()
    return {
        "avg": float(q.avg or 0),
        "p05": float(q.p05 or 0),
        "p50": float(q.p50 or 0),
        "p95": float(q.p95 or 0),
    }


# --------------------- geo heatmap ---------------------

@router.get("/geo/heatmap")
def geo_heatmap(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
    min_count: int = Query(1, ge=1),
):
    start_dt, end_dt = _start_date(days, start, end)
    q = (
        _media_base(db, start_dt, end_dt, media_type)
        .with_entities(
            dbm.Media.geohash6,
            func.count().label("c"),
            func.max(dbm.Media.created_at).label("latest"),
        )
        .filter(dbm.Media.geohash6.isnot(None))
        .group_by(dbm.Media.geohash6)
        .having(func.count() >= min_count)
        .order_by(desc("c"))
    )
    rows = q.all()
    return [
        {"geohash6": gh, "count": int(c), "latest": latest.isoformat()}
        for gh, c, latest in rows
    ]


# --------------------- geo hotspots (richer) ---------------------

@router.get("/geo/hotspots")
def geo_hotspots(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
    precision: int = Query(6, ge=4, le=6, description="Geohash precision (4–6, since we store gh6)"),
    min_count: int = Query(1, ge=1),
    limit: int = Query(200, ge=1, le=1000),
):

    start_dt, end_dt = _start_date(days, start, end)
    prev_end = start_dt - timedelta(microseconds=1)
    prev_start = prev_end - (end_dt - start_dt)

    geocol = func.substr(dbm.Media.geohash6, 1, precision)

    cur_rows = (
        _media_base(db, start_dt, end_dt, media_type)
        .with_entities(
            geocol.label("gh"),
            func.count().label("c"),
            func.max(dbm.Media.created_at).label("latest"),
            func.count(func.distinct(dbm.Media.user_username)).label("uploaders"),
        )
        .filter(dbm.Media.geohash6.isnot(None))
        .group_by(geocol)
        .having(func.count() >= min_count)
    ).all()

    # previous window counts
    prev_rows = (
        _media_base(db, prev_start, prev_end, media_type)
        .with_entities(geocol.label("gh"), func.count().label("c"))
        .filter(dbm.Media.geohash6.isnot(None))
        .group_by(geocol)
    ).all()
    prev_map = {str(gh): int(c) for gh, c in prev_rows}

    sev_rows = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            geocol.label("gh"),
            func.sum(case((dbm.Detection.severity == dbm.Severity.low, 1), else_=0)).label("low"),
            func.sum(case((dbm.Detection.severity == dbm.Severity.medium, 1), else_=0)).label("medium"),
            func.sum(case((dbm.Detection.severity == dbm.Severity.high, 1), else_=0)).label("high"),
        )
        .filter(dbm.Media.geohash6.isnot(None))
        .group_by(geocol)
    ).all()
    sev_map = {
        str(gh): {"low": int(low or 0), "medium": int(med or 0), "high": int(h or 0)}
        for gh, low, med, h in sev_rows
    }

    class_rows = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(geocol.label("gh"), dbm.Detection.class_name, func.count().label("c"))
        .filter(dbm.Media.geohash6.isnot(None))
        .group_by(geocol, dbm.Detection.class_name)
        .order_by(geocol, desc("c"))
    ).all()
    top_map: dict[str, list[tuple[str, int]]] = {}
    for gh, name, c in class_rows:
        key = str(gh)
        arr = top_map.setdefault(key, [])
        if len(arr) < 3:
            arr.append((name, int(c)))

    out = []
    for gh, c, latest, uploaders in cur_rows:
        gh = str(gh)
        c = int(c)
        prev = prev_map.get(gh, 0)
        trend_pct = ((c - prev) / prev) * 100.0 if prev > 0 else (100.0 if c > 0 else 0.0)

        lat = lon = None
        bbox = None
        if _geohash_mod:
            try:
                lat_, lon_, lat_err, lon_err = _geohash_mod.decode_exactly(gh)
                lat, lon = float(lat_), float(lon_)
                bbox = [lat - lat_err, lon - lon_err, lat + lat_err, lon + lon_err]
            except Exception:
                pass

        out.append({
            "geohash": gh,
            "precision": precision,
            "count": c,
            "prev_count": int(prev),
            "trend_pct": float(trend_pct),
            "latest": latest.isoformat() if latest else None,
            "uploaders": int(uploaders or 0),
            "severity": sev_map.get(gh, {"low": 0, "medium": 0, "high": 0}),
            "top_classes": [{"class_name": n, "count": cc} for n, cc in top_map.get(gh, [])],
            "lat": lat, "lon": lon,
            "bbox": bbox,  # [minLat, minLon, maxLat, maxLon]
        })

    out.sort(key=lambda r: r["count"], reverse=True)
    return out[:limit]


# --------------------- resolution time ---------------------

@router.get("/detections/time-to-resolution")
def time_to_resolution(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)
    delta_hours = func.extract("epoch", dbm.Detection.resolved_at - dbm.Detection.created_at) / 3600.0
    q = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            dbm.Detection.severity,
            func.avg(delta_hours).label("avg_hours"),
            func.percentile_cont(0.95).within_group(delta_hours).label("p95_hours"),
            func.count().label("n"),
        )
        .filter(dbm.Detection.resolved_at.isnot(None))
        .group_by(dbm.Detection.severity)
    )
    rows = q.all()
    return [
        {
            "severity": (sev.value if hasattr(sev, "value") else str(sev)),
            "avg_hours": float(avg or 0),
            "p95_hours": float(p95 or 0),
            "count": int(n),
        }
        for sev, avg, p95, n in rows
    ]

@router.get("/issues/aging-buckets")
def issues_aging_buckets(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
    include_unassigned: bool = Query(True, description="Include rows where assigned_to is NULL"),
    sla_high_h:   int = Query(24,  ge=1),
    sla_medium_h: int = Query(72,  ge=1),
    sla_low_h:    int = Query(168, ge=1),  # 7 days
    scope: Literal["backlog", "window"] = Query("backlog", description="backlog=all open issues (default), window=only issues whose media falls in the selected window"),
):

    start_dt, end_dt = _start_date(days, start, end)

    if scope == "window":
        base = (
            _detect_base(db, start_dt, end_dt, media_type)
            .filter(dbm.Detection.status == dbm.IssueStatus.open)
        )
    else:
        base = (
            db.query(dbm.Detection)
              .join(dbm.Frame, dbm.Detection.frame_id == dbm.Frame.id)
              .join(dbm.Media, dbm.Frame.media_id == dbm.Media.id)
              .filter(dbm.Detection.status == dbm.IssueStatus.open)
        )
        if media_type:
            base = base.filter(dbm.Media.media_type == media_type)
        if not include_unassigned:
            base = base.filter(dbm.Detection.assigned_to.isnot(None))

    if scope == "window" and not include_unassigned:
        base = base.filter(dbm.Detection.assigned_to.isnot(None))

    age_hours = func.extract("epoch", func.now() - dbm.Detection.created_at) / 3600.0

    bucket = case(
        (age_hours <= 24,  "0-24h"),
        (age_hours <= 72,  "1-3d"),
        (age_hours <= 168, "3-7d"),
        (age_hours <= 720, "7-30d"),
        else_=">30d",
    ).label("bucket")

    # -------- A) by severity (bucket counts) --------
    rows_sev = (
        base.with_entities(dbm.Detection.severity, bucket, func.count().label("c"))
            .group_by(dbm.Detection.severity, bucket)
            .all()
    )
    by_severity: dict[str, dict[str, int]] = {}
    for sev, b, c in rows_sev:
        sk = sev.value if hasattr(sev, "value") else str(sev)
        by_severity.setdefault(sk, {})
        by_severity[sk][str(b)] = int(c)

    open_counts = (
        base.with_entities(dbm.Detection.severity, func.count().label("c"))
            .group_by(dbm.Detection.severity)
            .all()
    )
    open_map = { (s.value if hasattr(s,"value") else str(s)): int(c) for s, c in open_counts }

    # -------- B) by assignee × severity --------
    rows_asg = (
        base.with_entities(
                dbm.Detection.assigned_to,
                dbm.Detection.severity,
                bucket,
                func.count().label("c"),
            )
            .group_by(dbm.Detection.assigned_to, dbm.Detection.severity, bucket)
            .order_by(dbm.Detection.assigned_to, dbm.Detection.severity)
            .all()
    )
    temp: dict[tuple[str | None, str], dict[str, int]] = {}
    for assignee, sev, b, c in rows_asg:
        sk = sev.value if hasattr(sev, "value") else str(sev)
        key = (assignee, sk)
        d = temp.setdefault(key, {})
        d[str(b)] = int(c)

    by_assignee: list[dict] = []
    for (assignee, sk), buckets in temp.items():
        total = int(sum(buckets.values()))
        by_assignee.append({
            "assignee": assignee,
            "severity": sk,
            "buckets": {
                "0-24h":  buckets.get("0-24h", 0),
                "1-3d":   buckets.get("1-3d", 0),
                "3-7d":   buckets.get("3-7d", 0),
                "7-30d":  buckets.get("7-30d", 0),
                ">30d":   buckets.get(">30d", 0),
            },
            "total": total,
        })

    # -------- C) SLA breach counts per severity --------
    breach_cond = (
        ((dbm.Detection.severity == dbm.Severity.high)   & (age_hours > sla_high_h)) |
        ((dbm.Detection.severity == dbm.Severity.medium) & (age_hours > sla_medium_h)) |
        ((dbm.Detection.severity == dbm.Severity.low)    & (age_hours > sla_low_h))
    )
    rows_breach = (
        base.with_entities(dbm.Detection.severity, func.count().label("c"))
            .filter(breach_cond)
            .group_by(dbm.Detection.severity)
            .all()
    )
    breach_map = { (s.value if hasattr(s,"value") else str(s)): int(c) for s, c in rows_breach }

    rate_map: dict[str, float] = {}
    for sev in ["high", "medium", "low"]:
        open_n   = float(open_map.get(sev, 0))
        breach_n = float(breach_map.get(sev, 0))
        rate_map[sev] = (breach_n / open_n * 100.0) if open_n > 0 else 0.0

    return {
        "window": {"start": start_dt.isoformat(), "end": end_dt.isoformat()},
        "sla_hours": {"low": sla_low_h, "medium": sla_medium_h, "high": sla_high_h},
        "by_severity": {
            k: {
                "0-24h":  v.get("0-24h", 0),
                "1-3d":   v.get("1-3d", 0),
                "3-7d":   v.get("3-7d", 0),
                "7-30d":  v.get("7-30d", 0),
                ">30d":   v.get(">30d", 0),
                "total":  int(sum(v.values())),
            } for k, v in by_severity.items()
        },
        "by_assignee": by_assignee,
        "sla_breach_open": breach_map,
        "sla_breach_rate": rate_map,
        "open_counts": open_map,
    }

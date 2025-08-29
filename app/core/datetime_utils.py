from datetime import datetime, timezone, date, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

RO_TZ = ZoneInfo('Europe/Bucharest')

def format_datetime_ro(dt: Optional[datetime], format_str: str = "%d %B %Y, %H:%M") -> Optional[str]:
    if dt is None:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    dt_ro = dt.astimezone(RO_TZ)
    
    return dt_ro.strftime(format_str)

def format_datetime_iso_ro(dt: Optional[datetime]) -> Optional[str]:

    if dt is None:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    dt_ro = dt.astimezone(RO_TZ)
    
    return dt_ro.isoformat()

def get_current_time_ro() -> datetime:
    return datetime.now(RO_TZ)

def get_today_ro():
    return get_current_time_ro().date()

def ro_date_bounds_to_utc(d: date) -> tuple[datetime, datetime]:

    start_ro = datetime.combine(d, time.min, tzinfo=RO_TZ)
    end_ro = datetime.combine(d, time.max, tzinfo=RO_TZ)
    return start_ro.astimezone(timezone.utc), end_ro.astimezone(timezone.utc)
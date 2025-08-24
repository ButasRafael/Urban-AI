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


# --------------------- model performance ---------------------

@router.get("/detections/confidence-by-class")
def confidence_by_class(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
    min_detections: int = Query(10, ge=1),
):
    start_dt, end_dt = _start_date(days, start, end)
    q = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            dbm.Detection.class_name,
            func.avg(dbm.Detection.confidence).label("avg_confidence"),
            func.stddev(dbm.Detection.confidence).label("std_confidence"),
            func.count().label("count"),
            func.percentile_cont(0.25).within_group(dbm.Detection.confidence).label("p25"),
            func.percentile_cont(0.75).within_group(dbm.Detection.confidence).label("p75"),
        )
        .group_by(dbm.Detection.class_name)
        .having(func.count() >= min_detections)
        .order_by(desc("avg_confidence"))
    )
    rows = q.all()
    return [
        {
            "class_name": name,
            "avg_confidence": float(avg or 0),
            "std_confidence": float(std or 0),
            "p25_confidence": float(p25 or 0),
            "p75_confidence": float(p75 or 0),
            "count": int(count),
            "reliability_score": float(avg or 0) * (1 - min(1, (std or 0) / max(0.01, avg or 0.01))) if avg else 0,
        }
        for name, avg, std, count, p25, p75 in rows
    ]


@router.get("/system/daily-health")
def daily_health_metrics(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)
    dcol = func.date(dbm.Media.created_at)
    
    # Daily upload and processing metrics
    upload_metrics = (
        _media_base(db, start_dt, end_dt, media_type)
        .with_entities(
            dcol.label("date"),
            func.count().label("total_uploads"),
            func.avg(dbm.Media.process_ms_total).label("avg_processing_ms"),
            func.percentile_cont(0.95).within_group(dbm.Media.process_ms_total).label("p95_processing_ms"),
        )
        .group_by(dcol)
        .order_by(dcol)
    ).all()

    detection_metrics = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            dcol.label("date"),
            func.count().label("total_detections"),
            func.avg(dbm.Detection.confidence).label("avg_confidence"),
            func.count(case((dbm.Detection.confidence >= 0.8, 1), else_=None)).label("high_confidence_count"),
        )
        .group_by(dcol)
        .order_by(dcol)
    ).all()

    upload_map = {str(date): (total, float(avg_ms or 0), float(p95_ms or 0)) 
                  for date, total, avg_ms, p95_ms in upload_metrics}
    detection_map = {str(date): (total, float(avg_conf or 0), int(high_conf or 0)) 
                     for date, total, avg_conf, high_conf in detection_metrics}
    
    all_dates = set(upload_map.keys()) | set(detection_map.keys())
    
    result = []
    for date_str in sorted(all_dates):
        up_total, up_avg_ms, up_p95_ms = upload_map.get(date_str, (0, 0, 0))
        det_total, det_avg_conf, det_high_conf = detection_map.get(date_str, (0, 0, 0))
        
        result.append({
            "date": date_str,
            "uploads": up_total,
            "detections": det_total,
            "avg_processing_ms": up_avg_ms,
            "p95_processing_ms": up_p95_ms,
            "avg_confidence": det_avg_conf,
            "high_confidence_rate": (det_high_conf / max(1, det_total)) * 100.0,
            "detections_per_upload": det_total / max(1, up_total),
        })
    
    return result


@router.get("/activity/user-engagement")
def user_engagement_metrics(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)

    user_stats = (
        _media_base(db, start_dt, end_dt, media_type)
        .with_entities(
            dbm.Media.user_username,
            func.count().label("total_uploads"),
            func.count(func.distinct(func.date(dbm.Media.created_at))).label("active_days"),
            func.min(dbm.Media.created_at).label("first_upload"),
            func.max(dbm.Media.created_at).label("last_upload"),
        )
        .group_by(dbm.Media.user_username)
        .order_by(desc("total_uploads"))
    ).all()
    
    window_days = (end_dt - start_dt).days + 1
    
    result = []
    for user, uploads, active_days, first, last in user_stats:
        activity_rate = (active_days / window_days) * 100.0 if window_days > 0 else 0
        uploads_per_day = uploads / max(1, active_days)

        consistency_score = min(100, activity_rate)
        frequency_score = min(100, uploads_per_day * 10)
        engagement_score = (consistency_score * 0.6 + frequency_score * 0.4)

        avg_hours_between = 0
        if uploads > 1 and first and last:
            total_hours = (last - first).total_seconds() / 3600.0
            avg_hours_between = total_hours / max(1, uploads - 1)
        
        result.append({
            "user": user,
            "total_uploads": uploads,
            "active_days": active_days,
            "activity_rate_pct": activity_rate,
            "uploads_per_active_day": uploads_per_day,
            "avg_hours_between_uploads": avg_hours_between,
            "engagement_score": engagement_score,
            "first_upload": first.isoformat() if first else None,
            "last_upload": last.isoformat() if last else None,
        })
    
    return result


@router.get("/activity/temporal-patterns")
def temporal_patterns(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)

    hourly_pattern = (
        _media_base(db, start_dt, end_dt, media_type)
        .with_entities(
            func.extract("hour", dbm.Media.created_at).label("hour"),
            func.count().label("uploads"),
            func.avg(dbm.Media.process_ms_total).label("avg_processing_ms"),
        )
        .group_by(func.extract("hour", dbm.Media.created_at))
        .order_by("hour")
    ).all()

    dow_pattern = (
        _media_base(db, start_dt, end_dt, media_type)
        .with_entities(
            func.extract("dow", dbm.Media.created_at).label("dow"),
            func.count().label("uploads"),
            func.avg(dbm.Media.process_ms_total).label("avg_processing_ms"),
        )
        .group_by(func.extract("dow", dbm.Media.created_at))
        .order_by("dow")
    ).all()
    
    # Map day numbers to names
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    return {
        "hourly": [
            {
                "hour": int(hour),
                "uploads": int(uploads),
                "avg_processing_ms": float(avg_ms or 0),
            }
            for hour, uploads, avg_ms in hourly_pattern
        ],
        "daily": [
            {
                "day_of_week": int(dow),
                "day_name": dow_names[int(dow)],
                "uploads": int(uploads),
                "avg_processing_ms": float(avg_ms or 0),
            }
            for dow, uploads, avg_ms in dow_pattern
        ],
    }


@router.get("/performance/resolution-efficiency")
def resolution_efficiency_metrics(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)

    resolution_stats = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            dbm.Detection.assigned_to,
            func.count().label("total_assigned"),
            func.count(case((dbm.Detection.resolved_at.isnot(None), 1), else_=None)).label("resolved_count"),
            func.avg(func.extract("epoch", dbm.Detection.resolved_at - dbm.Detection.created_at) / 3600.0).label("avg_resolution_hours"),
            func.percentile_cont(0.95).within_group(func.extract("epoch", dbm.Detection.resolved_at - dbm.Detection.created_at) / 3600.0).label("p95_resolution_hours"),
        )
        .filter(dbm.Detection.assigned_to.isnot(None))
        .group_by(dbm.Detection.assigned_to)
        .order_by(desc("total_assigned"))
    ).all()

    verification_stats = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            dbm.Detection.verified_by,
            func.count().label("verified_count"),
            func.avg(func.extract("epoch", dbm.Detection.verified_at - dbm.Detection.created_at) / 3600.0).label("avg_verification_lag_hours"),
        )
        .filter(dbm.Detection.verified_by.isnot(None))
        .filter(dbm.Detection.verified_at.isnot(None))
        .group_by(dbm.Detection.verified_by)
    ).all()

    workload_stats = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            dbm.Detection.assigned_to,
            func.count().label("assigned_count"),
            func.count(case((dbm.Detection.status == dbm.IssueStatus.open, 1), else_=None)).label("open_count"),
            func.count(case((dbm.Detection.status == dbm.IssueStatus.resolved, 1), else_=None)).label("resolved_count"),
            func.count(case((dbm.Detection.status == dbm.IssueStatus.ignored, 1), else_=None)).label("ignored_count"),
        )
        .group_by(dbm.Detection.assigned_to)
    ).all()

    assignee_performance = []
    for assigned_to, total_assigned, resolved_count, avg_hours, p95_hours in resolution_stats:
        resolution_rate = (resolved_count / max(1, total_assigned)) * 100.0
        
        assignee_performance.append({
            "assignee": assigned_to,
            "total_assigned": total_assigned,
            "resolved_count": resolved_count or 0,
            "resolution_rate_pct": resolution_rate,
            "avg_resolution_hours": float(avg_hours or 0),
            "p95_resolution_hours": float(p95_hours or 0),
        })
    
    verification_performance = [
        {
            "verifier": verified_by,
            "verified_count": verified_count,
            "avg_verification_lag_hours": float(avg_lag or 0),
        }
        for verified_by, verified_count, avg_lag in verification_stats
    ]
    
    workload_distribution = [
        {
            "assignee": assigned_to or "(unassigned)",
            "total_assigned": assigned_count,
            "open": open_count or 0,
            "resolved": resolved_count or 0,
            "ignored": ignored_count or 0,
            "completion_rate": (((resolved_count or 0) + (ignored_count or 0)) / max(1, assigned_count)) * 100.0,
        }
        for assigned_to, assigned_count, open_count, resolved_count, ignored_count in workload_stats
    ]
    
    return {
        "assignee_performance": assignee_performance,
        "verification_performance": verification_performance,
        "workload_distribution": sorted(workload_distribution, key=lambda x: x["total_assigned"], reverse=True),
    }


@router.get("/quality/detection-quality")
def detection_quality_metrics(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
):
    start_dt, end_dt = _start_date(days, start, end)

    quality_overview = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            func.count().label("total_detections"),
            func.count(case((dbm.Detection.status == dbm.IssueStatus.ignored, 1), else_=None)).label("ignored_count"),
            func.count(case((dbm.Detection.verified_by.isnot(None), 1), else_=None)).label("verified_count"),
            func.avg(dbm.Detection.confidence).label("avg_confidence"),
            func.avg(case((dbm.Detection.status == dbm.IssueStatus.ignored, dbm.Detection.confidence), else_=None)).label("avg_ignored_confidence"),
            func.avg(case((dbm.Detection.status == dbm.IssueStatus.resolved, dbm.Detection.confidence), else_=None)).label("avg_resolved_confidence"),
        )
    ).first()

    confidence_quality = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            case(
                (dbm.Detection.confidence >= 0.9, "0.9-1.0"),
                (dbm.Detection.confidence >= 0.8, "0.8-0.9"),
                (dbm.Detection.confidence >= 0.7, "0.7-0.8"),
                (dbm.Detection.confidence >= 0.6, "0.6-0.7"),
                else_="<0.6"
            ).label("confidence_range"),
            func.count().label("total_count"),
            func.count(case((dbm.Detection.status == dbm.IssueStatus.ignored, 1), else_=None)).label("ignored_count"),
            func.count(case((dbm.Detection.status == dbm.IssueStatus.resolved, 1), else_=None)).label("resolved_count"),
        )
        .group_by("confidence_range")
        .order_by("confidence_range")
    ).all()

    override_by_class = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            dbm.Detection.class_name,
            func.count().label("total_count"),
            func.count(case((dbm.Detection.verified_by.isnot(None), 1), else_=None)).label("verified_count"),
            func.count(case((dbm.Detection.status == dbm.IssueStatus.ignored, 1), else_=None)).label("ignored_count"),
            func.avg(dbm.Detection.confidence).label("avg_confidence"),
        )
        .group_by(dbm.Detection.class_name)
        .having(func.count() >= 5)
        .order_by(desc(func.count(case((dbm.Detection.status == dbm.IssueStatus.ignored, 1), else_=None))))
    ).all()

    daily_quality = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            func.date(dbm.Media.created_at).label("date"),
            func.count().label("total_detections"),
            func.count(case((dbm.Detection.status == dbm.IssueStatus.ignored, 1), else_=None)).label("ignored_count"),
            func.avg(dbm.Detection.confidence).label("avg_confidence"),
        )
        .group_by(func.date(dbm.Media.created_at))
        .order_by("date")
    ).all()

    total_detections = quality_overview.total_detections or 0
    ignored_count = quality_overview.ignored_count or 0
    verified_count = quality_overview.verified_count or 0
    
    false_positive_rate = (ignored_count / max(1, total_detections)) * 100.0
    verification_rate = (verified_count / max(1, total_detections)) * 100.0
    
    return {
        "overview": {
            "total_detections": total_detections,
            "false_positive_rate": false_positive_rate,
            "verification_rate": verification_rate,
            "avg_confidence": float(quality_overview.avg_confidence or 0),
            "avg_ignored_confidence": float(quality_overview.avg_ignored_confidence or 0),
            "avg_resolved_confidence": float(quality_overview.avg_resolved_confidence or 0),
        },
        "by_confidence_range": [
            {
                "confidence_range": conf_range,
                "total_count": total_count,
                "ignored_count": ignored_count or 0,
                "resolved_count": resolved_count or 0,
                "false_positive_rate": ((ignored_count or 0) / max(1, total_count)) * 100.0,
            }
            for conf_range, total_count, ignored_count, resolved_count in confidence_quality
        ],
        "by_class": [
            {
                "class_name": class_name,
                "total_count": total_count,
                "verified_count": verified_count or 0,
                "ignored_count": ignored_count or 0,
                "avg_confidence": float(avg_confidence or 0),
                "false_positive_rate": ((ignored_count or 0) / max(1, total_count)) * 100.0,
                "verification_rate": ((verified_count or 0) / max(1, total_count)) * 100.0,
            }
            for class_name, total_count, verified_count, ignored_count, avg_confidence in override_by_class
        ],
        "daily_trends": [
            {
                "date": str(date_val),
                "total_detections": total_detections,
                "ignored_count": ignored_count or 0,
                "false_positive_rate": ((ignored_count or 0) / max(1, total_detections)) * 100.0,
                "avg_confidence": float(avg_confidence or 0),
            }
            for date_val, total_detections, ignored_count, avg_confidence in daily_quality
        ]
    }


@router.get("/geography/issue-patterns")
def geographic_issue_patterns(
    db: Session = Depends(get_db),
    days: int | None = Query(None, ge=1, le=365),
    start: Optional[date] = None,
    end:   Optional[date] = None,
    media_type: Optional[Literal["image","video"]] = None,
    precision: int = Query(5, ge=4, le=6),
):
    start_dt, end_dt = _start_date(days, start, end)
    geocol = func.substr(dbm.Media.geohash6, 1, precision)

    geo_clusters = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            geocol.label("geohash"),
            func.count().label("total_issues"),
            func.count(func.distinct(dbm.Media.id)).label("unique_locations"),
            func.avg(func.extract("epoch", dbm.Detection.resolved_at - dbm.Detection.created_at) / 3600.0).label("avg_resolution_hours"),
            func.count(case((dbm.Detection.severity == dbm.Severity.high, 1), else_=None)).label("high_severity"),
            func.count(case((dbm.Detection.severity == dbm.Severity.medium, 1), else_=None)).label("medium_severity"),
            func.count(case((dbm.Detection.severity == dbm.Severity.low, 1), else_=None)).label("low_severity"),
            func.count(case((dbm.Detection.status == dbm.IssueStatus.resolved, 1), else_=None)).label("resolved_count"),
            func.max(dbm.Media.created_at).label("latest_issue"),
        )
        .filter(dbm.Media.geohash6.isnot(None))
        .group_by(geocol)
        .having(func.count() >= 2)
        .order_by(desc("total_issues"))
    ).all()

    recurring_locations = (
        _detect_base(db, start_dt, end_dt, media_type)
        .with_entities(
            geocol.label("geohash"),
            dbm.Detection.class_name,
            func.count().label("issue_count"),
            func.count(func.distinct(func.date(dbm.Media.created_at))).label("unique_days"),
            func.min(dbm.Media.created_at).label("first_occurrence"),
            func.max(dbm.Media.created_at).label("latest_occurrence"),
            func.avg(dbm.Detection.confidence).label("avg_confidence"),
        )
        .filter(dbm.Media.geohash6.isnot(None))
        .group_by(geocol, dbm.Detection.class_name)
        .having(func.count() >= 3)
        .having(func.count(func.distinct(func.date(dbm.Media.created_at))) >= 2)
        .order_by(desc("issue_count"))
    ).all()

    geo_performance = []
    for geohash, total_issues, unique_locations, avg_resolution_hours, high_sev, med_sev, low_sev, resolved_count, latest_issue in geo_clusters:
        resolution_rate = (resolved_count / max(1, total_issues)) * 100.0
        issue_density = total_issues / max(1, unique_locations)

        lat = lon = None
        if _geohash_mod and geohash:
            try:
                lat, lon = _geohash_mod.decode(str(geohash))
            except:
                pass
        
        geo_performance.append({
            "geohash": str(geohash),
            "total_issues": total_issues,
            "unique_locations": unique_locations,
            "issue_density": round(issue_density, 2),
            "avg_resolution_hours": float(avg_resolution_hours or 0),
            "resolution_rate": resolution_rate,
            "severity_distribution": {
                "high": high_sev or 0,
                "medium": med_sev or 0,
                "low": low_sev or 0,
            },
            "latest_issue": latest_issue.isoformat() if latest_issue else None,
            "lat": lat,
            "lon": lon,
        })
    
    recurring_patterns = [
        {
            "geohash": str(geohash),
            "class_name": class_name,
            "issue_count": issue_count,
            "unique_days": unique_days,
            "recurrence_rate": round(issue_count / max(1, unique_days), 2),
            "first_occurrence": first_occurrence.isoformat() if first_occurrence else None,
            "latest_occurrence": latest_occurrence.isoformat() if latest_occurrence else None,
            "avg_confidence": float(avg_confidence or 0),
        }
        for geohash, class_name, issue_count, unique_days, first_occurrence, latest_occurrence, avg_confidence in recurring_locations
    ]
    
    return {
        "geographic_clusters": geo_performance[:20],  # Top 20 problem areas
        "recurring_patterns": recurring_patterns[:15],  # Top 15 recurring issues
        "summary": {
            "total_problem_areas": len(geo_performance),
            "total_recurring_patterns": len(recurring_patterns),
            "avg_issues_per_area": sum(gp["total_issues"] for gp in geo_performance) / max(1, len(geo_performance)),
        }
    }


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

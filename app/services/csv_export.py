from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import Request
import csv
import io

_CSV_CACHE: dict[str, dict] = {}
_CSV_CACHE_EXPIRY = timedelta(minutes=15)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _abs_static_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/static/{filename}"


def image_url_for_media(media, request: Request, track_id: int | None = None) -> str | None:

    if not media:
        return None

    static_root = Path(os.getenv("STATIC_DIR", "static"))

    # 1) Track-specific thumbnail for videos
    if track_id is not None and getattr(media, "media_type", None) == "video":
        tracks_folder = f"{media.id}_tracks"
        track_thumb = f"track_{track_id}.jpg"
        track_path = static_root / tracks_folder / track_thumb
        if track_path.exists():
            return _abs_static_url(request, f"{tracks_folder}/{track_thumb}")

    # 2) UUID-based static filename (new system)
    static_filename = getattr(media, "static_filename", None)
    if static_filename and (static_root / static_filename).exists():
        return _abs_static_url(request, static_filename)

    # 3) Fallback to annotated image <id>.jpg
    annotated = f"{media.id}.jpg"
    if (static_root / annotated).exists():
        return _abs_static_url(request, annotated)

    # 4) Fallback to stored filename (normalize ext)
    filename = getattr(media, "filename", "")
    if filename:
        name = Path(filename).name
        ext = Path(name).suffix.lower()
        if ext not in IMAGE_EXTS:
            name = f"{Path(name).stem}.jpg"
        if (static_root / name).exists():
            return _abs_static_url(request, name)

    # 5) Last resort: return whatever static_filename we have, or annotated name, as an absolute URL
    if static_filename:
        return _abs_static_url(request, static_filename)
    return _abs_static_url(request, annotated)


def prepare_csv_data(chunks, request: Request) -> str:

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            'id', 'media_id', 'address', 'class_name', 'severity',
            'status', 'assigned_to', 'verified_by', 'description',
            'uploaded_at', 'resolved_at', 'verified_at', 'image_url'
        ],
    )
    writer.writeheader()

    for c in chunks:
        try:
            uploaded_str = c.uploaded_at.strftime("%Y-%m-%d %H:%M") if c.uploaded_at else ""
            resolved_str = c.resolved_at.strftime("%Y-%m-%d %H:%M") if c.resolved_at else ""
            verified_str = c.verified_at.strftime("%Y-%m-%d %H:%M") if c.verified_at else ""
            img_url = ""
            try:
                if getattr(c, "media", None):
                    img_url = image_url_for_media(c.media, request, getattr(c, "track_id", None)) or ""
            except Exception:
                pass

            writer.writerow({
                'id': c.id,
                'media_id': c.media_id,
                'address': c.address or "",
                'class_name': c.class_name or "",
                'severity': c.severity.value if c.severity else "",
                'status': c.status.value if c.status else "",
                'assigned_to': c.assigned_to or "",
                'verified_by': c.verified_by or "",
                'description': c.chunk or "",
                'uploaded_at': uploaded_str,
                'resolved_at': resolved_str,
                'verified_at': verified_str,
                'image_url': img_url,
            })
        except Exception as e:
            # Skip broken rows rather than failing the whole CSV
            print(f"[csv_export] row {getattr(c,'id',None)} error: {e}")
            continue

    return output.getvalue()


def store_csv_data(csv_id: str, csv_data: str) -> None:
    _CSV_CACHE[csv_id] = {
        "data": csv_data,
        "expires_at": datetime.now() + _CSV_CACHE_EXPIRY
    }
    # Cleanup expired
    expired = [k for k, v in _CSV_CACHE.items() if v['expires_at'] < datetime.now()]
    for k in expired:
        _CSV_CACHE.pop(k, None)


def get_csv_data(csv_id: str) -> str | None:
    entry = _CSV_CACHE.get(csv_id)
    if not entry:
        return None
    if entry["expires_at"] < datetime.now():
        _CSV_CACHE.pop(csv_id, None)
        return None
    return entry["data"]

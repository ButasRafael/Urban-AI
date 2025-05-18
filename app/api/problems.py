# app/api/problems.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.security import require_roles
from app.models import media as dbm
from app.models.schemas_portal import ProblemOut          # see ยง1.2 below

router = APIRouter(prefix="/problems", tags=["Problems"],
                   dependencies=[require_roles("admin", "authority")])

@router.get("", response_model=List[ProblemOut])
def all_problems(
    media_type: str | None = Query(None, pattern="^(image|video)$"),
    klass: str | None = Query(None, description="Filter by YOLO/SAM class"),
    db: Session = Depends(get_db)
):
    q = db.query(dbm.Media)
    if media_type:
        q = q.filter(dbm.Media.media_type == media_type)
    rows = q.order_by(dbm.Media.created_at.desc()).all()

    out: list[ProblemOut] = []
    for m in rows:
        classes_q = (db.query(dbm.Detection.class_name)
                       .join(dbm.Frame, dbm.Frame.id == dbm.Detection.frame_id)
                       .filter(dbm.Frame.media_id == m.id)
                       .distinct())
        if klass:
            classes_q = classes_q.filter(dbm.Detection.class_name == klass)
        classes = [c[0] for c in classes_q]
        if klass and not classes:           # filter out if class not found
            continue

        detects = (
            db.query(dbm.Detection)
              .join(dbm.Frame, dbm.Frame.id == dbm.Detection.frame_id)
              .filter(dbm.Frame.media_id == m.id)
              .all()
        )
        descriptions = [d.description or "n/a" for d in detects]
        solutions    = [d.solution    or "n/a" for d in detects]
        out.append(ProblemOut(
            media_id=m.id,
            address=m.address,
            latitude=m.latitude,
            longitude=m.longitude,
            user_username=m.user_username,
            media_type=m.media_type,
            annotated_image_url = f"/static/{m.id}.jpg" if m.media_type=="image" else None,
            annotated_video_url = f"/static/{m.id}.mp4" if m.media_type=="video" else None,
            created_at=m.created_at,
            predicted_classes=classes,
            descriptions=descriptions,
            solutions=solutions,
        ))
    return out
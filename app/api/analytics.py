# app/api/analytics.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, timedelta
from app.core.database import get_db
from app.core.security import require_roles
from app.models import media as dbm

router = APIRouter(prefix="/analytics", tags=["Analytics"],
                   dependencies=[require_roles("admin")])

@router.get("/uploads-by-day")
def uploads_last_week(db: Session = Depends(get_db)):
    start = date.today() - timedelta(days=6)
    rows = (db.query(func.date(dbm.Media.created_at), func.count())
              .filter(dbm.Media.created_at >= start)
              .group_by(func.date(dbm.Media.created_at))
              .order_by(func.date(dbm.Media.created_at))
              .all())
    return [{"date": str(d), "count": c} for d, c in rows]

@router.get("/uploads-by-user")
def uploads_by_user(db: Session = Depends(get_db)):
    rows = (db.query(dbm.Media.user_username, func.count())
              .group_by(dbm.Media.user_username)
              .all())
    return [{"user": u, "count": c} for u, c in rows]
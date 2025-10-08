from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import List, Optional, Literal
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import require_roles
from app.models.user import User

router = APIRouter(
    prefix="/users",
    tags=["Users"],
    dependencies=[require_roles("user", "authority", "admin")],
)

class UserOut(BaseModel):
    username: str
    role: str

@router.get("", response_model=List[UserOut])
def search_users(
    q: str = Query("", description="Search by username (substring, empty for all)"),
    limit: int = Query(8, ge=1, le=50),
    role: Optional[Literal["user", "authority", "admin"]] = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    if q:  # Only filter by username if q is provided
        query = query.filter(User.username.ilike(f"%{q}%"))
    if role:
        query = query.filter(User.role == role)
    rows = query.order_by(User.username.asc()).limit(limit).all()
    return [UserOut(username=r.username, role=r.role) for r in rows]

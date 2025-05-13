import enum
from sqlalchemy import Column, String, DateTime, Enum as SQLEnum
from datetime import datetime, timezone
from app.core.database import Base
from sqlalchemy.orm import relationship

class RoleEnum(str, enum.Enum):
    user = "user"
    authority = "authority"
    admin = "admin"

class User(Base):
    __tablename__ = "users"

    username        = Column(String, primary_key=True, index=True)
    hashed_password = Column(String, nullable=False)
    role            = Column(
                        SQLEnum(RoleEnum, name="role_enum"),
                        default=RoleEnum.user,
                        nullable=False,
                     )
    created_at = Column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc)
    )
    uploads = relationship("Media", backref="uploader")


from sqlalchemy import Column, String, DateTime
from datetime import datetime, timezone
from app.core.database import Base

class RevokedToken(Base):
    __tablename__ = "revoked_tokens"

    jti        = Column(String, primary_key=True, index=True)
    revoked_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

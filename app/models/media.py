from __future__ import annotations
import enum
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text,
    Enum as SQLEnum, Index
)
from sqlalchemy.orm import relationship
from app.core.database import Base


class DetectionSource(str, enum.Enum):
    yolo         = "yolo"
    gpt_dino     = "gpt_dino"
    sam_fallback = "sam_fallback"


class IssueStatus(str, enum.Enum):
    open     = "open"
    resolved = "resolved"
    ignored  = "ignored"


class Severity(str, enum.Enum):
    low    = "low"
    medium = "medium"
    high   = "high"


class Media(Base):
    __tablename__ = "media"

    id            = Column(Integer, primary_key=True)
    filename      = Column(String, nullable=False)
    media_type    = Column(String, nullable=False)
    user_username = Column(
        String,
        ForeignKey("users.username", ondelete="CASCADE"),
        nullable=False,
    )

    width      = Column(Integer)
    height     = Column(Integer)
    num_frames = Column(Integer)

    address   = Column(String, nullable=True)
    latitude  = Column(Float,  nullable=True)
    longitude = Column(Float,  nullable=True)
    geohash6  = Column(String(12), nullable=True, index=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    process_ms_total = Column(Integer, nullable=True)

    rag_chunks = relationship("RAGChunk", back_populates="media", cascade="all,delete")
    frames     = relationship("Frame", back_populates="media", cascade="all,delete")


class Frame(Base):
    __tablename__ = "frame"

    id          = Column(Integer, primary_key=True)
    media_id    = Column(Integer, ForeignKey("media.id", ondelete="CASCADE"))
    frame_index = Column(Integer)
    timestamp   = Column(Float)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    media   = relationship("Media", back_populates="frames")
    detects = relationship("Detection", back_populates="frame", cascade="all,delete")


class Detection(Base):
    __tablename__ = "detection"

    id         = Column(Integer, primary_key=True)
    frame_id   = Column(Integer, ForeignKey("frame.id", ondelete="CASCADE"))
    track_id   = Column(Integer, nullable=True)

    class_id   = Column(Integer, nullable=False)
    class_name = Column(String,  nullable=False, index=True)
    confidence = Column(Float)

    x1 = Column(Float); y1 = Column(Float); x2 = Column(Float); y2 = Column(Float)
    mask_rle     = Column(JSON)
    mask_polygon = Column(JSON)

    description  = Column(Text)
    solution     = Column(Text)

    source     = Column(
        SQLEnum(DetectionSource, name="detection_source_enum"),
        nullable=False,
        default=DetectionSource.yolo,
    )
    status     = Column(
        SQLEnum(IssueStatus, name="issue_status_enum"),
        nullable=False,
        default=IssueStatus.open,
    )
    severity   = Column(
        SQLEnum(Severity, name="severity_enum"),
        nullable=False,
        default=Severity.medium,
    )
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    assigned_to = Column(String, ForeignKey("users.username", ondelete="SET NULL"), nullable=True)
    verified_by = Column(String, ForeignKey("users.username", ondelete="SET NULL"), nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)

    frame = relationship("Frame", back_populates="detects")

    __table_args__ = (
        Index("ix_detection_class_name_created_at", "class_name", "created_at"),
    )

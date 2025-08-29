from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Text, ForeignKey, DateTime, String, Float, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy import Enum as SQLEnum
from pgvector.sqlalchemy import Vector
from app.core.database import Base
from app.models.media import IssueStatus, Severity

class RAGChunk(Base):
    __tablename__ = "rag_chunks"

    id        = Column(Integer, primary_key=True)
    media_id  = Column(Integer, ForeignKey("media.id", ondelete="CASCADE"), nullable=False)
    track_id  = Column(Integer, nullable=True)
    detection_id = Column(Integer, ForeignKey("detection.id", ondelete="CASCADE"), nullable=True)
    
    # Core content
    chunk     = Column(Text, nullable=False)
    embedding = Column(Vector(3072), nullable=False)
    
    # Static fields (embedded in vector, won't change)
    address   = Column(String, nullable=True)
    media_type = Column(String, nullable=True)
    class_name = Column(String, nullable=True, index=True)
    uploaded_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Dynamic fields (NOT in embedding, can change)
    severity = Column(SQLEnum(Severity, name="severity_enum"), nullable=True, index=True)
    status = Column(SQLEnum(IssueStatus, name="issue_status_enum"), nullable=True, index=True)
    assigned_to = Column(String, nullable=True, index=True)
    verified_by = Column(String, nullable=True, index=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True, index=True)
    verified_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Additional metadata as JSON for flexibility
    extra_metadata = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), 
                        onupdate=lambda: datetime.now(timezone.utc))
    
    media = relationship("Media", back_populates="rag_chunks")
    
    __table_args__ = (
        Index("ix_rag_chunk_severity_status", "severity", "status"),
        Index("ix_rag_chunk_class_created", "class_name", "created_at"),
    )

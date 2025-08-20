from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.core.database import Base

class RAGChunk(Base):
    __tablename__ = "rag_chunks"

    id        = Column(Integer, primary_key=True)
    media_id  = Column(Integer, ForeignKey("media.id", ondelete="CASCADE"), nullable=False)
    chunk     = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    media = relationship("Media", back_populates="rag_chunks")

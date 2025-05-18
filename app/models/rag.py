from sqlalchemy import Column, Integer, Text, Float, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.core.database import Base


class RAGChunk(Base):
    """Denormalised natural‑language blob + embedding."""
    __tablename__ = "rag_chunks"

    id = Column(Integer, primary_key=True)
    media_id = Column(Integer, ForeignKey("media.id", ondelete="CASCADE"), nullable=False)
    chunk = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)

    media = relationship("Media", back_populates="rag_chunks")

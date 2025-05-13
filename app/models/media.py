from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from app.core.database import Base


class Media(Base):
    __tablename__ = "media"

    id         = Column(Integer, primary_key=True)
    filename   = Column(String, nullable=False)
    media_type = Column(String, nullable=False)
    user_username = Column(
      String,
      ForeignKey("users.username", ondelete="CASCADE"),
      nullable=False,
    )
    width      = Column(Integer)
    height     = Column(Integer)
    num_frames = Column(Integer)
    address    = Column(String, nullable=True)
    latitude   = Column(Float,  nullable=True)
    longitude  = Column(Float,  nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    frames = relationship("Frame", back_populates="media", cascade="all,delete")


class Frame(Base):
    __tablename__ = "frame"

    id          = Column(Integer, primary_key=True)
    media_id    = Column(Integer, ForeignKey("media.id", ondelete="CASCADE"))
    frame_index = Column(Integer)
    timestamp   = Column(Float)
    media   = relationship("Media", back_populates="frames")
    detects = relationship("Detection", back_populates="frame", cascade="all,delete")


class Detection(Base):
    __tablename__ = "detection"

    id           = Column(Integer, primary_key=True)
    frame_id     = Column(Integer, ForeignKey("frame.id", ondelete="CASCADE"))
    track_id     = Column(Integer, nullable=True)
    class_id     = Column(Integer, nullable=False)
    class_name   = Column(String, nullable=False)
    confidence   = Column(Float)
    x1           = Column(Float); y1 = Column(Float); x2 = Column(Float); y2 = Column(Float)
    mask_rle     = Column(JSON)
    mask_polygon = Column(JSON)
    description  = Column(Text) 
    solution     = Column(Text)

    frame = relationship("Frame", back_populates="detects")

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class Mask(BaseModel):
    rle: dict
    polygon: List[List[float]]


class Detection(BaseModel):
    track_id: Optional[int] = None
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    mask: Mask
    description: Optional[str] = None
    solution:    Optional[str] = None
    severity: Optional[Literal["low", "medium", "high"]] = "medium"
    source: Optional[Literal["yolo", "gpt_dino", "sam_fallback"]] = None


class FrameOut(BaseModel):
    frame_index: int
    timestamp_ms: float
    objects: List[Detection]


class ImageResponse(BaseModel):
    media_id: int
    annotated_image_url: str
    frames: List[FrameOut]
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    suggestions: List[Detection] = Field(default_factory=list)


class VideoResponse(BaseModel):
    media_id: int
    frames: List[FrameOut]
    annotated_video_url: str
    address: Optional[str] = None 
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class MediaListItem(BaseModel):
    media_id: int
    media_type: str
    annotated_image_url: Optional[str]
    annotated_video_url: Optional[str]
    created_at: datetime
    address: Optional[str]
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    predicted_classes: List[str] = Field(default_factory=list)
    descriptions: List[str] = Field(default_factory=list)
    solutions: List[str] = Field(default_factory=list)




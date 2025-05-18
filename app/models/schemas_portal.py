from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

class ProblemOut(BaseModel):
    media_id: int
    address: Optional[str]
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    user_username: str
    media_type: str
    annotated_image_url: Optional[str]
    annotated_video_url: Optional[str]
    created_at: datetime
    predicted_classes: List[str] = []
    descriptions: List[str] = []
    solutions: List[str] = []
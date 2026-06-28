from pydantic import BaseModel, Field
from typing import List, Optional

class Query(BaseModel):
    query: str
    top_k: int = 10
    filter: str = ""

class Clip(BaseModel):
    clip_url: Optional[str] = None
    clip_file_path: Optional[str] = None
    start_time: float = Field(description="The start time of the clip in seconds.")
    end_time: float = Field(description="The end time of the clip in seconds.")
    video_id: str
    id: str
    embedding: Optional[List[float]] = None

class Video(BaseModel):
    video_url_or_path: str
    audio_url: Optional[str]
    id: str
    clips: List[Clip]
    duration: int

class ClipRetrievalResult(BaseModel):
    clip: Clip
    distance: float
    video_id: str
    score: float
    collection_name: str

class Frame(BaseModel):
    frame_url: Optional[str] = None
    frame_file_path: str
    timestamp: float
    video_id: str
    id: str

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

    def __repr__(self) -> str:
        # Truncate clip_url: it may be a huge base64 data URL (range-proxy inline
        # clips), which would otherwise flood logs on any print/repr of a Clip.
        u = self.clip_url
        if u and len(u) > 64:
            u = f"{u[:48]}...<{len(u)} chars>"
        return (f"Clip(id={self.id!r}, video_id={self.video_id!r}, "
                f"start={self.start_time}, end={self.end_time}, "
                f"clip_url={u!r}, clip_file_path={self.clip_file_path!r})")

    __str__ = __repr__

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

    def __repr__(self) -> str:
        u = self.frame_url
        if u and len(u) > 64:
            u = f"{u[:48]}...<{len(u)} chars>"
        return (f"Frame(id={self.id!r}, video_id={self.video_id!r}, "
                f"timestamp={self.timestamp}, frame_url={u!r}, "
                f"frame_file_path={self.frame_file_path!r})")

    __str__ = __repr__

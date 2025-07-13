from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from api.v1.embeddings.models.common import ModelSelectionRequest, EmbeddingDetail
from api.v1.common import TraceInfo


class VideoEmbeddingRequest(ModelSelectionRequest):
    """
    Request model for generating video.py embeddings.
    """
    url: str = Field(..., description="URL to the video.py resource.")
    caption: Optional[str] = Field(None, description="Optional caption for the video.py.")
    tags: Optional[List[str]] = Field(default_factory=list, description="Optional tags for the video.py.")
    frame_sampling: Optional[str] = Field(None, description="Frame sampling strategy, e.g., 'uniform', 'scene'.")
    generate_summary: bool = Field(False, description="Whether to generate a summary for the video.py.")


class VideoEmbeddingResponse(BaseModel):
    """
    Response model for video embedding operations.
    Includes multiple types of embeddings and related metadata.
    """
    embeddings: Dict[str, EmbeddingDetail] = Field(
        ..., description="Embeddings grouped by type: 'content', 'visual', 'transcript', etc."
    )
    transcript_text: Optional[str] = Field(
        None, description="Transcript extracted from the video, if available."
    )
    summary: Optional[str] = Field(
        None, description="Optional summary generated from the transcript."
    )
    trace: Optional[TraceInfo] = Field(
        None, description="Trace information for this video embedding request."
    )

from fastapi import APIRouter
from api.v1.embeddings.models.video import VideoEmbeddingRequest, VideoEmbeddingResponse

router = APIRouter(tags=["Video"])


@router.post("/video", response_model=VideoEmbeddingResponse)
def embeddings_video(request: VideoEmbeddingRequest):
    """
    Endpoint for generating embeddings from video.py input.
    """
    return VideoEmbeddingResponse

from fastapi import APIRouter
from .text import router as text_router
from .aggregate import router as aggregate_router
from .video import router as video_router


"""
Embeddings adapters routes.
"""
embeddings_router = APIRouter(prefix="/embeddings", tags=["Embeddings"])
embeddings_router.include_router(text_router)
# embeddings_router.include_router(video_router)
embeddings_router.include_router(aggregate_router)

from fastapi import APIRouter
from api.v1.embeddings import embeddings_router
# from generation import generation_router
# from speech import speech_router

v1 = APIRouter(prefix="/v1", tags=["v1"])

v1.include_router(embeddings_router)
# v1.include_router(generation_router)
# v1.include_router(speech_router)

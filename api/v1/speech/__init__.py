from fastapi import APIRouter

speech_router = APIRouter(prefix="/text", tags=["Text", "Audio", "Speech"])
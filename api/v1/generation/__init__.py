from fastapi import APIRouter

generation_router = APIRouter(prefix="/generation", tags=["Generation", "Chat Completion"])

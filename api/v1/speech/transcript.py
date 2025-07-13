from fastapi import APIRouter
from api.v1.embeddings.models import TranscriptRequest, TranscriptResponse
from services.asr_wrapper import extract_transcript

router = APIRouter()

@router.post("/transcript", response_model=TranscriptResponse)
def transcript_endpoint(request: TranscriptRequest):
    return extract_transcript(request)
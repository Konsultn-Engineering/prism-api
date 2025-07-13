import datetime

from fastapi import APIRouter, Depends
from api.v1.embeddings.models.text import TextEmbeddingRequest, TextEmbeddingResponse  # your Pydantic D
from api.v1.embeddings.models.common import EmbeddingVector, ModelInfo
from api.v1.common import TraceInfo
from api.v1.embeddings.service import embedding_service
from core.services.embedding_service import EmbeddingService
from uuid import uuid4

router = APIRouter(tags=["Text"])


@router.post("/text", response_model=TextEmbeddingResponse)
def embeddings_text(request: TextEmbeddingRequest, svc: EmbeddingService = Depends(embedding_service)):
    embedding = svc.embed_text(
        model_name=request.model_name,
        text=request.text,
        contextual_focus=request.contextual_focus
    )

    return TextEmbeddingResponse(
        embedding=EmbeddingVector(
            vector=embedding.vector,
            dimension=embedding.model_info.embedding_dim,
        ),
        model=ModelInfo(
            name=embedding.model_info.name,
            backend=embedding.model_info.name
        ),
        trace=TraceInfo(
            trace_id=uuid4(),
            timestamp=datetime.datetime.utcnow(),
            latency_ms=embedding.latency_ms or 0.0,
        )
    )

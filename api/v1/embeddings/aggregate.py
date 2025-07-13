import datetime
from uuid import uuid4
from fastapi import APIRouter, Depends
from api.v1.embeddings.models.aggregate import AggregateRequest, AggregateResponse
from api.v1.common import TraceInfo
from core.services.embedding_service import EmbeddingService
from core.types import EmbeddingItem
from api.v1.embeddings.service import embedding_service

router = APIRouter(tags=["Aggregate"])


@router.post("/aggregate", response_model=AggregateResponse)
def embeddings_aggregate(request: AggregateRequest, svc: EmbeddingService = Depends(embedding_service)):
    """
    Endpoint for aggregating multiple embeddings.
    """
    embedding_items = [EmbeddingItem(embedding=item.embedding, weight=item.weight) for item in request.items]

    result = svc.aggregate_embeddings(weighted_embeddings=embedding_items, strategy=request.strategy, query=request.query)

    return AggregateResponse(
        embedding=result.vector,
        strategy=request.strategy,
        trace=TraceInfo(
            trace_id=uuid4(),
            timestamp=datetime.datetime.utcnow(),
            latency_ms=result.latency_ms
        )
    )
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from api.v1.embeddings.models.common import BaseEmbeddingRequest
from api.v1.common import TraceInfo


class AggregateItem(BaseModel):
    """
    Single item for aggregation operations.
    """
    embedding: List[float] = Field(..., description="Embedding vector.")
    weight: float = Field(1.0, description="Weight assigned to this embedding.")


class AggregateRequest(BaseEmbeddingRequest):
    """
    Request model for aggregating multiple embeddings.
    """
    items: List[AggregateItem] = Field(..., description="List of embedding items to aggregate.")
    strategy: str = Field("mean", description="Aggregation strategy, e.g., 'mean', 'weighted_mean', 'attention'.")
    query: Optional[str] = Field("", description="Optional: Query text for attention-based aggregation strategies.")
    model_name: Optional[str] = Field(
        None, description="Optional: Name of the model to use for this aggregation operation."
    )


class AggregateResponse(BaseModel):
    """
    Response model for aggregated embedding.
    """
    embedding: List[float] = Field(..., description="Detailed information about the aggregated embedding.")
    strategy: str = Field(..., description="Aggregation strategy used.")
    trace: TraceInfo = Field(..., description="Trace details for the current operation.")

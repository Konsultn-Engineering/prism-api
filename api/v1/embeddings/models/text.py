from pydantic import BaseModel, Field
from api.v1.embeddings.models.common import ModelSelectionRequest, EmbeddingVector
from api.v1.common import ModelInfo, TraceInfo


class TextEmbeddingRequest(ModelSelectionRequest):
    """
    Request model for generating text embeddings.
    """
    text: str = Field(..., description="Input text to generate an embedding for.")


class TextEmbeddingResponse(BaseModel):
    embedding: EmbeddingVector = Field(..., description="Embedding vector and dimension.")
    model: ModelInfo = Field(..., description="Metadata about the model used.")
    trace: TraceInfo = Field(..., description="Trace metadata for this request.")

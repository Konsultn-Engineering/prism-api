from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from api.v1.common import ModelInfo


class ContextualFocus(BaseModel):
    """
    Represents a term and its associated weight for contextual focus in embeddings.
    """
    term: str = Field(..., description="Contextual term to focus on.")
    weight: float = Field(..., description="Weight associated with the contextual term.")


class BaseEmbeddingRequest(BaseModel):
    """
    Base request model for embedding operations that may utilize contextual focus.
    """
    contextual_focus: Optional[List[ContextualFocus]] = Field(
        default_factory=list, description="List of contextual focus terms and weights."
    )


class ModelSelectionRequest(BaseEmbeddingRequest):
    """
    Adds model selection capability to requests.
    """
    model_name: Optional[str] = Field(
        None, description="Optional: Name of the model to use for this embedding operation."
    )


class EmbeddingVector(BaseModel):
    vector: List[float] = Field(..., description="The raw embedding vector.")
    dimension: int = Field(..., description="The dimensionality of the embedding vector.")


class EmbeddingDetail(BaseModel):
    vector: List[float] = Field(..., description="Embedding vector.")
    dimension: int = Field(..., description="Vector dimensionality.")
    model: ModelInfo = Field(..., description="Model information for this embedding.")

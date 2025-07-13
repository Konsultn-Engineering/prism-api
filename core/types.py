from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass


class EmbeddingItem(TypedDict):
    embedding: List[float]
    weight: float


@dataclass
class ModelInfo:
    """Common model information structure."""
    name: str
    embedding_dim: int
    max_length: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingOperation:
    vector: List[float]
    model_info: Optional[ModelInfo] = None
    latency_ms: Optional[float] = None

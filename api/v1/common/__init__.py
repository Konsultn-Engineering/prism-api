from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from uuid import UUID


class ModelInfo(BaseModel):
    """
    Metadata about the model used for encoding.
    """
    name: str = Field(..., description="Model name or version identifier.")
    backend: Optional[str] = Field(None,
                                   description="Model backend or framework (e.g. 'clip', 'sentence-transformers').")


class TraceInfo(BaseModel):
    """
    Metadata used for tracing, debugging, or observability.
    """
    trace_id: UUID = Field(..., description="Unique trace identifier for this request.")
    timestamp: datetime = Field(..., description="Timestamp of the embedding request.")
    latency_ms: Optional[float] = Field(None, description="Time taken to complete the operation in milliseconds.")
    source: Optional[str] = Field(None, description="Origin service or source tag of the request.")

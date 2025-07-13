from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class ModelInfo:
    """Common model information structure."""
    name: str
    embedding_dim: int
    max_length: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def __getitem__(self, key: str) -> Any:
        """Make ModelInfo subscript for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        elif self.metadata and key in self.metadata:
            return self.metadata[key]
        else:
            raise KeyError(f"Key '{key}' not found in ModelInfo")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "embedding_dim": self.embedding_dim,
        }
        if self.max_length is not None:
            result["max_length"] = self.max_length
        if self.metadata:
            result.update(self.metadata)
        return result


@dataclass
class TranscriptionSegment:
    """Individual segment of transcription."""
    text: str
    start: float
    end: float
    confidence: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    segments: Optional[List[TranscriptionSegment]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StorageObject:
    """Metadata for a storage object."""
    key: str
    size: int
    last_modified: datetime
    etag: Optional[str] = None
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class StorageLocation:
    """Reference to a storage location."""
    bucket: str
    key: str
    url: Optional[str] = None

    @property
    def path(self) -> str:
        """Get the full path (bucket/key)."""
        return f"{self.bucket}/{self.key}"

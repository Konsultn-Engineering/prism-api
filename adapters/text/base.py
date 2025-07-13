# adapters/text/base.py
from abc import ABC, abstractmethod
from typing import Union, List, Optional, Dict, Literal, Any
import numpy as np
from adapters.types import ModelInfo


class TextEncoder(ABC):
    """Abstract interface for text encoding models."""
    @abstractmethod
    def with_model(self, model_name: str) -> "ContextualTextEncoder":
        pass

    @abstractmethod
    def encode(
            self,
            text: Union[str, List[str]],
            normalize: bool = True,
            **kwargs
    ) -> np.ndarray:
        """Encode text(s) to embeddings."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimension of embeddings produced."""
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return model information including name, dimensions, etc."""
        pass


class ContextualTextEncoder(TextEncoder):
    """Extended interface for text encoders with chunking and contextual focus."""

    @abstractmethod
    def encode(
            self,
            text: Union[str, List[str]],
            normalize: bool = True,
            strategy: Optional[Literal["auto", "truncate", "mean_pool", "max_pool", "stride"]] = "auto",
            chunk_size: Optional[int] = None,
            stride: int = 50,
            contextual_focus: Optional[Union[List[str], Dict[str, float]]] = None,
            focus_weight: float = 2.0,
            **kwargs
    ) -> np.ndarray:
        """Encode with chunking and contextual focus support."""
        pass

    @abstractmethod
    def encode_with_context_weighting(
            self,
            text: str,
            context_weights: Dict[str, float],
            normalize: bool = True
    ) -> np.ndarray:
        """Encode with sophisticated context weighting."""
        pass

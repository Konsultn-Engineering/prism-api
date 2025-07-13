from abc import ABC, abstractmethod
from typing import Union, List, Optional
import numpy as np
from PIL import Image
from adapters.types import ModelInfo


class ImageEncoder(ABC):
    """Abstract interface for image encoding models."""

    @abstractmethod
    def encode_image(
            self,
            image: Union[str, Image.Image, np.ndarray],
            normalize: bool = True
    ) -> np.ndarray:
        """Encode a single image into an embedding vector."""
        pass

    @abstractmethod
    def encode_images(
            self,
            images: List[Union[str, Image.Image, np.ndarray]],
            batch_size: Optional[int] = None,
            aggregate: str = "none",
            normalize: bool = True
    ) -> np.ndarray:
        """Encode multiple images."""
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the model."""
        pass


class MultiModalEncoder(ImageEncoder):  # ← Fixed: removed redundant ABC
    """Interface for models that can encode both images and text."""

    @abstractmethod
    def encode_text(
            self,
            text: Union[str, List[str]],
            aggregate: str = "none",
            normalize: bool = True,
            **kwargs
    ) -> np.ndarray:
        """Encode text(s) into embeddings."""
        pass

    @abstractmethod
    def compute_similarity(
            self,
            embeddings1: np.ndarray,
            embeddings2: np.ndarray,
            type1: str = "auto",
            type2: str = "auto"
    ) -> np.ndarray:
        """Compute similarity between two sets of embeddings."""
        pass
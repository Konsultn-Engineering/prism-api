from functools import lru_cache
from typing import List, Union, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from adapters.vision.base import MultiModalEncoder
from adapters.types import ModelInfo

_DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=4)
def get_clip_model_and_processor(model_name: str = _DEFAULT_CLIP_MODEL) -> Tuple[CLIPModel, CLIPProcessor]:
    """Load and cache CLIP model and processor."""
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    return model, processor


class CLIPAdapter(MultiModalEncoder):
    """
    Adapter for CLIP (Contrastive Language-Image Pre-training) model.
    Provides unified interface for image and text embeddings.
    """

    def __init__(self, model_name: str = _DEFAULT_CLIP_MODEL, device: Optional[str] = None):
        self._model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = get_clip_model_and_processor(self._model_name)
        # Move model to the specified device
        self.model = model.to(self.device)
        self.processor = processor

    def encode_image(
            self,
            image: Union[str, Image.Image, np.ndarray],
            normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a single image into CLIP embedding.

        Args:
            image: Path to image file, PIL Image, or numpy array
            normalize: Whether to L2 normalize the output

        Returns:
            Embedding vector
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().squeeze()

    def encode_images(
            self,
            images: List[Union[str, Image.Image, np.ndarray]],
            batch_size: Optional[int] = None,
            aggregate: str = "none",
            normalize: bool = True
    ) -> np.ndarray:
        """
        Encode multiple images.

        Args:
            images: List of image paths, PIL Images, or numpy arrays
            batch_size: Process images in batches (if None, process all at once)
            aggregate: How to combine embeddings ('mean', 'none')
            normalize: Whether to L2 normalize outputs

        Returns:
            If aggregate='mean': Single embedding (mean of all)
            If aggregate='none': Array of shape (n_images, embedding_dim)
        """
        if batch_size is None:
            batch_size = len(images)

        all_features = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            pil_images = []

            for img in batch:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img).convert("RGB")
                pil_images.append(img)

            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)

            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)

            all_features.append(features.cpu())

        # Concatenate all batches
        all_features = torch.cat(all_features, dim=0).numpy()

        if aggregate == "mean":
            return all_features.mean(axis=0)
        elif aggregate == "none":
            return all_features
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")

    def encode_text(
            self,
            text: Union[str, List[str]],
            aggregate: str = "none",
            normalize: bool = True,
            **kwargs
    ) -> np.ndarray:
        """
        Encode text(s) into CLIP embeddings.

        Args:
            text: Single text or list of texts
            aggregate: How to combine if multiple texts ('mean', 'none')
            normalize: Whether to L2 normalize outputs
            **kwargs: Additional parameters (for interface compatibility)

        Returns:
            If single text: 1D embedding
            If multiple texts and aggregate='none': 2D array
            If multiple texts and aggregate='mean': 1D embedding
        """
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_features = text_features.cpu().numpy()

        if is_single:
            return text_features.squeeze()
        elif aggregate == "mean":
            return text_features.mean(axis=0)
        else:
            return text_features

    def compute_similarity(
            self,
            embeddings1: np.ndarray,
            embeddings2: np.ndarray,
            type1: str = "auto",
            type2: str = "auto"
    ) -> np.ndarray:
        """
        Compute cosine similarity between embeddings.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            type1: Type hint for first embeddings (unused in CLIP)
            type2: Type hint for second embeddings (unused in CLIP)

        Returns:
            Similarity matrix
        """
        # Ensure 2D arrays
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)

        # Convert to tensors if needed
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)

        # Compute cosine similarity
        similarity = torch.matmul(embeddings1, embeddings2.T)
        return similarity.numpy()

    def get_model_info(self) -> ModelInfo:
        """Return model information."""
        return ModelInfo(
            name=self._model_name,
            embedding_dim=self.model.config.projection_dim,
            max_length=self.model.config.text_config.max_position_embeddings,
            metadata={
                "vision_model": self.model.config.vision_config.model_type,
                "text_model": self.model.config.text_config.model_type,
                "vision_embedding_dim": self.model.config.vision_config.hidden_size,
                "text_embedding_dim": self.model.config.text_config.hidden_size,
            }
        )

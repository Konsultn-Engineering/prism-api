# adapters/text/sentence_transformer.py
from functools import lru_cache
from typing import List, Union, Optional, Literal, Dict
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from adapters.text.base import ContextualTextEncoder
from adapters.types import ModelInfo

_DEFAULT_ST_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def get_sentence_transformer(model_name: str = _DEFAULT_ST_MODEL) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


class SentenceTransformerAdapter(ContextualTextEncoder):
    """
    Adapter for SentenceTransformer models with chunking and contextual focus support.
    """

    def get_embedding_dim(self) -> int:
        pass

    def __init__(self, model_name: str = _DEFAULT_ST_MODEL):
        self._model_name = model_name
        self.model = get_sentence_transformer(self._model_name)
        self._max_seq_length = self.model.max_seq_length
        self.tokenizer = self.model.tokenizer

    def with_model(self, model_name: str) -> ContextualTextEncoder:
        if model_name and model_name != self._model_name:
            return SentenceTransformerAdapter(model_name)
        else:
            return self

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
        """
        Unified encoding method with automatic chunking and contextual focus.

        Args:
            text: Single text or list of texts to encode
            normalize: Whether to L2-normalize embeddings
            strategy: Chunking strategy
                - "auto": Automatically decide based on text length (default)
                - "truncate": Use only first chunk
                - "mean_pool": Average all chunk embeddings
                - "max_pool": Element-wise max across chunks
                - "stride": Overlapping chunks
            chunk_size: Tokens per chunk (defaults to model max)
            stride: Overlap size for stride strategy
            contextual_focus: Terms to emphasize. Can be:
                - List[str]: Each term gets focus_weight
                - Dict[str, float]: Term -> weight mapping
            focus_weight: Default weight for focus terms (if list provided)

        Returns:
            Embeddings array (1D for single text, 2D for multiple)
        """
        # Handle batch processing
        if isinstance(text, list):
            embeddings = []
            for t in text:
                emb = self.encode(
                    t,
                    normalize=normalize,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    stride=stride,
                    contextual_focus=contextual_focus,
                    focus_weight=focus_weight
                )
                embeddings.append(emb)
            return np.array(embeddings)

        # Apply contextual focus if provided
        if contextual_focus:
            text = self._apply_contextual_focus(text, contextual_focus, focus_weight)

        # Determine strategy
        if strategy == "auto":
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) <= self._max_seq_length:
                # Text fits, no chunking needed
                return self.model.encode(text, normalize_embeddings=normalize)
            else:
                # Default to mean_pool for long texts
                strategy = "mean_pool"

        # If no chunking needed
        if strategy not in ["truncate", "mean_pool", "max_pool", "stride"]:
            return self.model.encode(text, normalize_embeddings=normalize)

        # Apply chunking strategy
        return self._encode_with_chunking(
            text,
            strategy,
            chunk_size or self._max_seq_length,
            stride,
            normalize
        )

    @classmethod
    def _apply_contextual_focus(
            cls,
            text: str,
            contextual_focus: Union[List[str], Dict[str, float]],
            default_weight: float
    ) -> str:
        """
        Apply contextual focus by strategically repeating focus terms.

        The approach:
        1. Append focus terms at the end (maintains readability)
        2. Weight determines repetition count
        3. Interleave terms to avoid clustering
        """
        # Normalize to dict format
        if isinstance(contextual_focus, list):
            focus_dict = {term: default_weight for term in contextual_focus}
        else:
            focus_dict = contextual_focus

        # Build focus suffix
        focus_parts = []
        for term, weight in focus_dict.items():
            # Repetition based on weight (e.g., weight=2.0 → repeat 2 times)
            repeat_count = max(1, int(weight))
            for _ in range(repeat_count):
                focus_parts.append(term)

        # Shuffle to avoid clustering same terms
        import random
        random.shuffle(focus_parts)

        # Append focus terms with a separator
        if focus_parts:
            focus_suffix = " [CONTEXT] " + " ".join(focus_parts)
            return text + focus_suffix

        return text

    def _encode_with_chunking(
            self,
            text: str,
            strategy: Literal["truncate", "mean_pool", "max_pool", "stride"],
            chunk_size: int,
            stride: int,
            normalize: bool
    ) -> np.ndarray:
        """Handle text chunking with specified strategy."""
        tokens = self.tokenizer.tokenize(text)

        if strategy == "truncate":
            truncated_tokens = tokens[:chunk_size]
            truncated_text = self.tokenizer.convert_tokens_to_string(truncated_tokens)
            return self.model.encode(truncated_text, normalize_embeddings=normalize)

        # Create chunks
        chunks = []
        if strategy == "stride":
            for i in range(0, len(tokens), chunk_size - stride):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)
                if i + chunk_size >= len(tokens):
                    break
        else:
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)

        # Encode all chunks
        embeddings = self.model.encode(chunks, normalize_embeddings=False)

        result: np.ndarray = np.array([])

        # Aggregate
        if strategy in ["mean_pool", "stride"]:
            result = np.mean(embeddings, axis=0)
        elif strategy == "max_pool":
            result = np.max(embeddings, axis=0)

        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm

        return result

    def encode_with_context_weighting(
            self,
            text: str,
            context_weights: Dict[str, float],
            normalize: bool = True
    ) -> np.ndarray:
        """
        Alternative approach: Weighted combination of embeddings.
        More sophisticated than repetition-based focusing.
        """
        # Base embedding
        base_embedding = self.model.encode(text, normalize_embeddings=False)

        # Context embeddings
        context_embeddings = []
        weights: List[float] = []

        for term, weight in context_weights.items():
            # Encode the term in context
            context_text = f"{text} [SEP] {term}"
            emb = self.model.encode(context_text, normalize_embeddings=False)
            context_embeddings.append(emb)
            weights.append(weight)

        # Weighted combination
        if context_embeddings:
            context_embeddings = np.array(context_embeddings)
            weights: np.ndarray = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights

            # Combine: 70% base + 30% weighted contexts
            context_component = np.average(context_embeddings, axis=0, weights=weights)
            final_embedding = 0.7 * base_embedding + 0.3 * context_component
        else:
            final_embedding = base_embedding

        # Normalize
        if normalize:
            norm = np.linalg.norm(final_embedding)
            if norm > 0:
                final_embedding = final_embedding / norm

        return final_embedding

    def get_model_info(self) -> ModelInfo:
        """Return model information."""
        return ModelInfo(
            name=self._model_name,
            embedding_dim=self.model.get_sentence_embedding_dimension(),
            max_length=self._max_seq_length,
        )

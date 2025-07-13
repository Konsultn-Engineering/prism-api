"""
Supports: weighted mean, max/mean/norm pooling, PCA denoising, and query-aware attention with interaction weights.
"""

from typing import List, Optional, Dict, cast
import numpy as np
from core.types import EmbeddingItem


class EmbeddingAggregator:
    """
    Aggregates a list of weighted embeddings using configurable strategies.
    Stateless; safe for concurrent use.
    """
    _STRATEGY_DOCS: Dict[str, Dict[str, str]] = {
        "weighted_mean": {
            "desc": "Weighted mean using provided weights. Use for baseline aggregation when you trust supplied "
                    "weights.",
            "best_for": "Most general purpose; when you have confidence in the weights (e.g., engagement, recency, "
                        "importance)."
        },
        "max_pool": {
            "desc": "Elementwise maximum across all vectors. Highlights strong activations and rare features.",
            "best_for": "Highlighting rare but strong features; e.g., detection of outlier or highly-activated "
                        "concepts."
        },
        "norm_weighted": {
            "desc": "Uses L2 norm of each embedding as its weight (confidence). Useful when embeddings vary in "
                    "magnitude.",
            "best_for": "Variable-quality sources; when higher-norm vectors are more reliable or signal strength "
                        "matters."
        },
        "pca_denoised": {
            "desc": "Removes dominant direction (first principal component) from weighted mean. SIF-style denoising.",
            "best_for": "Long-form or multi-sentence fusion; denoises common background signal for better semantic "
                        "focus."
        },
        "query_attention": {
            "desc": "Attention-weighted mean: combines interaction weight and relevance to a query vector (softmax "
                    "over weighted cosine scores).",
            "best_for": "Personalized fusion; when you want to bias aggregation toward a user/query/intent vector."
        },
    }

    def aggregate(
            self,
            items: List[EmbeddingItem],
            strategy: str = "weighted_mean",
            query: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Aggregate a list of weighted embeddings using the given strategy. Args: items: List of dicts with
        'embedding': List[float], 'weight': float strategy: Aggregation strategy name query: Optional[List[float]] (
        required for 'query_attention') contextual_focus: Optional[List[Dict]] (contextual focus objects,
        currently unused but available for advanced aggregation) Returns: Aggregated embedding as List[float]
        """
        if not items:
            raise ValueError("No embeddings provided for aggregation.")

        matrix: np.ndarray = np.array([v["embedding"] for v in items], dtype=np.float32)
        weights: np.ndarray = np.array([v["weight"] for v in items], dtype=np.float32)

        if matrix.ndim != 2 or matrix.shape[0] != len(weights):
            raise ValueError("Invalid input dimensions.")

        if strategy == "weighted_mean":
            return self._weighted_mean(matrix, weights)
        elif strategy == "max_pool":
            return self._max_pool(matrix)
        elif strategy == "norm_weighted":
            return self._norm_weighted(matrix)
        elif strategy == "pca_denoised":
            return self._pca_denoised(matrix, weights)
        elif strategy == "query_attention":
            if query is None:
                raise ValueError("query is required for query_attention strategy")
            return self._query_attention(matrix, query, weights)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    @classmethod
    def available_strategies(cls) -> List[Dict[str, str]]:
        """
        Returns a list of available strategies with their descriptions and best-use cases.
        """
        return [
            {
                "name": name,
                "description": doc["desc"],
                "best_used_for": doc["best_for"]
            }
            for name, doc in cls._STRATEGY_DOCS.items()
        ]

    @classmethod
    def strategies_as_dict(cls) -> Dict[str, Dict[str, str]]:
        """
        Returns a dict mapping strategy name to its details.
        """
        return {name: doc for name, doc in cls._STRATEGY_DOCS.items()}

    # ---- Aggregation Implementations ----

    @staticmethod
    def _weighted_mean(matrix: np.ndarray, weights: np.ndarray) -> List[float]:
        weights = weights / (np.sum(weights) + 1e-8)
        result: np.ndarray = np.average(matrix, axis=0, weights=weights)
        return EmbeddingAggregator.to_list(result)

    @staticmethod
    def _max_pool(matrix: np.ndarray) -> List[float]:
        result: np.ndarray = np.max(matrix, axis=0)
        return EmbeddingAggregator.to_list(result)

    @staticmethod
    def _norm_weighted(matrix: np.ndarray) -> List[float]:
        norms: np.ndarray = np.linalg.norm(matrix, axis=1)
        weights: np.ndarray = norms / (np.sum(norms) + 1e-8)
        result: np.ndarray = np.average(matrix, axis=0, weights=weights)
        return EmbeddingAggregator.to_list(result)

    def _pca_denoised(self, matrix: np.ndarray, weights: np.ndarray) -> List[float]:
        mean_vec: np.ndarray = np.array(self._weighted_mean(matrix, weights))
        centered: np.ndarray = matrix - mean_vec
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        pc1: np.ndarray = vt[0]
        projection: np.ndarray = mean_vec - np.dot(mean_vec, pc1) * pc1
        return EmbeddingAggregator.to_list(projection)

    @staticmethod
    def _query_attention(matrix: np.ndarray, query: List[float], weights: np.ndarray) -> List[float]:
        q: np.ndarray = np.array(query, dtype=np.float32)
        if q.shape[0] != matrix.shape[1]:
            raise ValueError("Query vector and embeddings must have the same dimension.")

        norms: np.ndarray = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        sims: np.ndarray = np.dot(matrix, q) / (norms + 1e-8)
        combined_weights: np.ndarray = weights * sims
        attention_scores: np.ndarray = np.exp(combined_weights - np.max(combined_weights))
        attention_scores = attention_scores / np.sum(attention_scores)
        result: np.ndarray = np.average(matrix, axis=0, weights=attention_scores)
        return EmbeddingAggregator.to_list(result)

    @staticmethod
    def to_list(matrix: np.ndarray) -> List[float]:
        return cast(List[float], matrix.tolist())

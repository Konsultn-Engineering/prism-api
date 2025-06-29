"""
embedding.py

Core embedding utilities and model loading functions for the embed-server project.

This module provides efficient, cached loading of transformer models (text, CLIP, Whisper),
along with utilities for chunked text embedding and weighted averaging of embedding vectors.
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

@lru_cache()
def get_text_model():
    """
    Load and cache the SentenceTransformer model for text embeddings.

    Returns:
        SentenceTransformer: The loaded text embedding model, moved to CUDA if available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

@lru_cache()
def get_clip_model():
    """
    Load and cache the CLIPModel for visual embeddings.

    Returns:
        CLIPModel: The loaded CLIP visual embedding model.
    """
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

@lru_cache()
def get_clip_processor():
    """
    Load and cache the CLIPProcessor for image preprocessing.

    Returns:
        CLIPProcessor: The loaded CLIP image preprocessor.
    """
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@lru_cache()
def get_whisper_model():
    """
    Load and cache the Whisper ASR model for audio transcription.

    Returns:
        whisper.Whisper: The loaded Whisper model (base variant).
    """
    import whisper
    return whisper.load_model("base")

def chunk_and_embed(text: str) -> np.ndarray:
    """
    Embeds text, automatically chunking if sequence exceeds the model's max token length.

    Args:
        text (str): The input text to embed.

    Returns:
        np.ndarray: The embedding vector for the input text (average across chunks if chunked).
    """
    model = get_text_model()
    tokenizer = model.tokenizer
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= 512:
        return model.encode(text)
    chunks = [tokenizer.convert_tokens_to_string(tokens[i:i + 512]) for i in range(0, len(tokens), 512)]
    return np.mean([model.encode(chunk) for chunk in chunks], axis=0)

def weighted_average(vectors: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Computes the weighted average of a list of embedding vectors.

    Args:
        vectors (list[np.ndarray]): List of embedding vectors (same shape).
        weights (list[float]): Corresponding weights for each vector.

    Returns:
        np.ndarray: Weighted average embedding vector.
    """
    weights = np.array(weights).reshape(-1, 1)
    stacked = np.vstack(vectors)
    return np.sum(stacked * weights, axis=0) / np.sum(weights)
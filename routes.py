"""
routes.py

This module defines the API endpoints for the embed-server project, including text, video, 
and aggregate embedding endpoints. It relies on modular utilities and model interfaces for clean, 
efficient request handling.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import tempfile
import os
import numpy as np

from .models import (
    TextQuery, EmbedResponse, VideoEmbedResponse,
    PreferenceRequest, PreferenceResponse
)
from .embedding import (
    chunk_and_embed, weighted_average,
    get_clip_model, get_clip_processor, get_whisper_model
)
from .video_utils import extract_audio, extract_frames

router = APIRouter()

@router.post("/embed/text", response_model=EmbedResponse)
async def embed_text(q: TextQuery):
    """
    Generate a text embedding for the provided input.

    Args:
        q (TextQuery): The request body containing the input text.

    Returns:
        EmbedResponse: The resulting embedding vector.
    """
    try:
        vector = chunk_and_embed(q.text)
        return {"embedding": vector.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed/video", response_model=VideoEmbedResponse)
async def embed_video(
    file: UploadFile = File(...),
    caption: str = Form(""),
    tags: str = Form(""),
    frame_mode: str = Form("uniform")
):
    """
    Generate visual and textual embeddings from a video file, with optional caption and tags.

    Args:
        file (UploadFile): Uploaded video file.
        caption (str, optional): Optional caption describing the video.
        tags (str, optional): Optional tags for the video content.
        frame_mode (str, optional): Frame extraction mode ("uniform" or "scene").

    Returns:
        VideoEmbedResponse: Contains video, content, and transcript embeddings, plus transcript text.
    """
    tmp_path = None
    audio_path = None
    frame_paths = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        # Transcribe audio
        audio_path = extract_audio(tmp_path)
        asr_model = get_whisper_model()
        transcript = asr_model.transcribe(audio_path)["text"]

        # Frame extraction
        frame_paths = extract_frames(tmp_path, mode=frame_mode)
        images = [Image.open(p).convert("RGB") for p in frame_paths]

        # Visual embedding using CLIP
        processor = get_clip_processor()
        model = get_clip_model()
        inputs = processor(images=images, return_tensors="pt", padding=True)
        features = model.get_image_features(**inputs).detach().numpy()
        video_embedding = np.mean(features, axis=0)

        # Transcript embedding
        transcript_embedding = chunk_and_embed(transcript)

        # Content embedding (caption + tags + transcript)
        full_text = f"{caption} {tags} {transcript}".strip()
        content_embedding = chunk_and_embed(full_text)

        return VideoEmbedResponse(
            video_embedding=video_embedding.tolist(),
            content_embedding=content_embedding.tolist(),
            transcript_embedding=transcript_embedding.tolist(),
            transcript=transcript
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        for p in frame_paths:
            if os.path.exists(p):
                os.remove(p)

@router.post("/embed/aggregate", response_model=PreferenceResponse)
async def embed_user_preference(body: PreferenceRequest):
    """
    Compute a weighted aggregation of user preference embedding vectors.

    Args:
        body (PreferenceRequest): List of vectors and corresponding weights.

    Returns:
        PreferenceResponse: Weighted aggregate embedding vector.
    """
    try:
        visual_vectors = []
        visual_weights = []
        text_vectors = []
        text_weights = []

        for item in body.vectors:
            vec = np.array(item.embedding)
            if vec.shape[0] == 512:
                visual_vectors.append(vec)
                visual_weights.append(item.weight)
            elif vec.shape[0] == 384:
                text_vectors.append(vec)
                text_weights.append(item.weight)
            else:
                raise HTTPException(status_code=400, detail="Unsupported embedding dimensionality")

        components = []
        mod_weights = []

        if text_vectors:
            text_pref = weighted_average(text_vectors, text_weights)
            components.append(text_pref)
            mod_weights.append(0.7)

        if visual_vectors:
            visual_pref_512 = weighted_average(visual_vectors, visual_weights)
            # TODO: replace with learned projection later
            projected = np.matmul(visual_pref_512.reshape(1, -1), np.random.rand(512, 384)).flatten()
            components.append(projected)
            mod_weights.append(0.3)

        if not components:
            raise HTTPException(status_code=400, detail="No valid vectors provided")

        final_embedding = weighted_average(components, mod_weights)
        return PreferenceResponse(embedding=final_embedding.tolist())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
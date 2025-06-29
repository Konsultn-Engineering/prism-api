from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import tempfile
import os
import numpy as np

from .models import (
    TextQuery, EmbedResponse, VideoEmbedResponse,
    PreferenceItem, PreferenceRequest, PreferenceResponse
)
from .embedding import chunk_and_embed, weighted_average
from .video_utils import extract_audio, extract_frames

router = APIRouter()

@router.post("/embed/text", response_model=EmbedResponse)
async def embed_text(q: TextQuery):
    try:
        vector = chunk_and_embed(q.text)
        return {"embedding": vector.tolist()}
    except Exception:
        raise HTTPException(status_code=500)

@router.post("/embed/video", response_model=VideoEmbedResponse)
async def embed_video(
    file: UploadFile = File(...),
    caption: str = Form(""),
    tags: str = Form(""),
    frame_mode: str = Form("uniform")
):
    import whisper
    from transformers import CLIPProcessor, CLIPModel

    # Lazy-load models
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    asr_model = whisper.load_model("base")

    tmp_path = None
    audio_path = None
    frame_paths = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        # Transcription
        audio_path = extract_audio(tmp_path)
        transcript = asr_model.transcribe(audio_path)["text"]

        # Frame extraction
        frame_paths = extract_frames(tmp_path, mode=frame_mode)
        images = [Image.open(p).convert("RGB") for p in frame_paths]

        # Visual embedding using CLIP
        inputs = processor(images=images, return_tensors="pt", padding=True)
        features = model.get_image_features(**inputs).detach().numpy()
        video_embedding = np.mean(features, axis=0)

        # Transcript embedding (pure spoken words)
        transcript_embedding = chunk_and_embed(transcript)

        # Content embedding (caption + tags + transcript)
        full_text = f"{caption} {tags} {transcript}".strip()
        content_embedding = chunk_and_embed(full_text)

        return {
            "video_embedding": video_embedding.tolist(),
            "content_embedding": content_embedding.tolist(),
            "transcript_embedding": transcript_embedding.tolist(),
            "transcript": transcript
        }
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
        return {"embedding": final_embedding.tolist()}

    except Exception:
        raise HTTPException(status_code=500)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import tempfile
import ffmpeg
from PIL import Image
from io import BytesIO
import os
from functools import lru_cache

app = FastAPI()

# === Models with lazy loading and device placement === #
@lru_cache()
def get_text_model():
    # Load sentence transformer model with GPU support if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

@lru_cache()
def get_clip_model():
    # Load CLIP visual model
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

@lru_cache()
def get_clip_processor():
    # Load processor for CLIP model (handles image preprocessing)
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Utility Functions === #
def chunk_and_embed(text: str) -> np.ndarray:
    # Split long text into <=512-token chunks and average their embeddings
    model = get_text_model()
    tokenizer = model.tokenizer
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= 512:
        return model.encode(text)
    chunks = [tokenizer.convert_tokens_to_string(tokens[i:i + 512]) for i in range(0, len(tokens), 512)]
    return np.mean([model.encode(chunk) for chunk in chunks], axis=0)

def extract_audio(video_path: str) -> str:
    # Convert video file to WAV format for transcription
    audio_path = video_path.replace(".mp4", ".wav")
    ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000').overwrite_output().run(quiet=True)
    return audio_path

def extract_frames(video_path: str, mode: str = "uniform", max_frames: int = 5) -> list[str]:
    # Extract representative frames from the video
    frame_dir = tempfile.mkdtemp()
    pattern = os.path.join(frame_dir, "frame_%03d.jpg")

    # Choose frame extraction strategy
    if mode == "scene":
        vf_expr = "select='gt(scene,0.3)',showinfo"
        vsync = "vfr"
    else:
        vf_expr = f"select=not(mod(n\\,{30}))"
        vsync = None

    # Run ffmpeg frame extraction
    ffmpeg.input(video_path).output(
        pattern,
        vf=vf_expr,
        vsync=vsync,
        vframes=max_frames
    ).overwrite_output().run(quiet=True)

    # Return paths of extracted frames (up to max_frames)
    return [os.path.join(frame_dir, f) for f in sorted(os.listdir(frame_dir))[:max_frames]]

def fuse_vectors(text_vec: np.ndarray, vis_vec: np.ndarray, alpha=0.7, beta=0.3) -> np.ndarray:
    # Weighted average of text and visual embeddings
    return (alpha * text_vec + beta * vis_vec)

# === Response Model === #
class EmbedResponse(BaseModel):
    embedding: list[float]           # Final output embedding vector
    transcript: str | None = None    # Optional transcript
    frame_mode: str | None = None    # Optional frame extraction mode

# === Text Embedding Endpoint === #
class TextQuery(BaseModel):
    text: str

@app.post("/embed/text", response_model=EmbedResponse)
async def embed_text(q: TextQuery):
    try:
        # Embed input text (chunked if necessary)
        vector = chunk_and_embed(q.text)
        return {"embedding": vector.tolist()}
    except Exception:
        raise HTTPException(status_code=500)

# === Video Embedding Endpoint === #
@app.post("/embed/video", response_model=EmbedResponse)
async def embed_video(
    file: UploadFile = File(...),
    caption: str = Form(""),
    tags: str = Form(""),
    frame_mode: str = Form("uniform")
):
    if file.content_type not in ("video/mp4",):
        raise HTTPException(status_code=400)

    # Lazy import Whisper model
    import whisper
    asr_model = whisper.load_model("base")

    tmp_path = None
    audio_path = None
    frame_paths = []
    try:
        # Save uploaded video file to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        # Extract audio and transcribe using Whisper
        audio_path = extract_audio(tmp_path)
        transcript = asr_model.transcribe(audio_path)["text"]

        # Extract multiple representative frames
        frame_paths = extract_frames(tmp_path, mode=frame_mode)
        images = [Image.open(p).convert("RGB") for p in frame_paths]

        # Encode visual content using CLIP
        processor = get_clip_processor()
        model = get_clip_model()
        inputs = processor(images=images, return_tensors="pt", padding=True)
        features = model.get_image_features(**inputs).detach().numpy()
        visual_vector = np.mean(features, axis=0)

        # Encode combined text content
        combined_text = f"{caption} {tags} {transcript}".strip()
        text_vector = chunk_and_embed(combined_text)

        # Fuse embeddings
        fused = fuse_vectors(text_vector, visual_vector).tolist()

        return {"embedding": fused, "transcript": transcript, "frame_mode": frame_mode}

    finally:
        # Cleanup temp audio and video
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        for p in frame_paths:
            if os.path.exists(p):
                os.remove(p)

# === Image Embedding Endpoint === #
@app.post("/embed/images", response_model=EmbedResponse)
async def embed_images(
    files: list[UploadFile] = File(...),
    caption: str = Form(""),
    tags: str = Form("")
):
    try:
        # Convert uploaded images to RGB
        images = [Image.open(BytesIO(await f.read())).convert("RGB") for f in files]
        processor = get_clip_processor()
        model = get_clip_model()

        # Encode images with CLIP
        inputs = processor(images=images, return_tensors="pt", padding=True)
        features = model.get_image_features(**inputs).detach().numpy()
        image_vector = np.mean(features, axis=0)

        # Optionally combine with text
        if caption or tags:
            text_vector = chunk_and_embed(f"{caption} {tags}")
            fused = fuse_vectors(text_vector, image_vector).tolist()
        else:
            fused = image_vector.tolist()

        return {"embedding": fused}
    except Exception:
        raise HTTPException(status_code=500)

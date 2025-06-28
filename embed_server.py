from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

# === Lazy-Loaded Models === #
@lru_cache()
def get_text_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

@lru_cache()
def get_clip_model():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

@lru_cache()
def get_clip_processor():
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Utility Functions === #
def chunk_and_embed(text: str) -> np.ndarray:
    model = get_text_model()
    tokenizer = model.tokenizer
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= 512:
        return model.encode(text)
    chunks = [tokenizer.convert_tokens_to_string(tokens[i:i + 512]) for i in range(0, len(tokens), 512)]
    return np.mean([model.encode(chunk) for chunk in chunks], axis=0)

def extract_audio(video_path: str) -> str:
    audio_path = video_path.replace(".mp4", ".wav")
    ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000').overwrite_output().run(quiet=True)
    return audio_path

def extract_frames(video_path: str, mode: str = "uniform", max_frames: int = 5) -> list[str]:
    frame_dir = tempfile.mkdtemp()
    pattern = os.path.join(frame_dir, "frame_%03d.jpg")
    if mode == "scene":
        vf_expr = "select='gt(scene,0.3)',showinfo"
        vsync = "vfr"
    else:
        vf_expr = f"select=not(mod(n\\,{30}))"
        vsync = None
    ffmpeg.input(video_path).output(
        pattern,
        vf=vf_expr,
        vsync=vsync,
        vframes=max_frames
    ).overwrite_output().run(quiet=True)
    return [os.path.join(frame_dir, f) for f in sorted(os.listdir(frame_dir))[:max_frames]]

def weighted_average(vectors: list[np.ndarray], weights: list[float]) -> np.ndarray:
    weights = np.array(weights).reshape(-1, 1)
    stacked = np.vstack(vectors)
    return np.sum(stacked * weights, axis=0) / np.sum(weights)

# === Models === #
class TextQuery(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list[float]

class VideoEmbedResponse(BaseModel):
    video_embedding: list[float]
    content_embedding: list[float]
    transcript_embedding: list[float]
    transcript: str

class PreferenceItem(BaseModel):
    embedding: list[float]
    weight: float

class PreferenceRequest(BaseModel):
    vectors: list[PreferenceItem]

class PreferenceResponse(BaseModel):
    embedding: list[float]

# === Endpoints === #
@app.post("/embed/text", response_model=EmbedResponse)
async def embed_text(q: TextQuery):
    try:
        vector = chunk_and_embed(q.text)
        return {"embedding": vector.tolist()}
    except Exception:
        raise HTTPException(status_code=500)


@app.post("/embed/video", response_model=dict)
async def embed_video(
    file: UploadFile = File(...),
    caption: str = Form(""),
    tags: str = Form(""),
    frame_mode: str = Form("uniform")
):
    import whisper
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
        processor = get_clip_processor()
        model = get_clip_model()
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


@app.post("/embed/aggregate", response_model=PreferenceResponse)
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

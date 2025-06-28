from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import whisper
import torch
import numpy as np
import tempfile
import ffmpeg
from PIL import Image
import os

app = FastAPI()

# Load models once
text_model = SentenceTransformer("all-MiniLM-L6-v2")
asr_model = whisper.load_model("base")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


### === UTILS === ###

def chunk_and_embed(text: str) -> np.ndarray:
    tokenizer = text_model.tokenizer
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= 512:
        return text_model.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), 512):
        chunk = tokenizer.convert_tokens_to_string(tokens[i:i + 512])
        chunks.append(text_model.encode(chunk))
    return np.mean(chunks, axis=0)

def extract_audio(video_path: str) -> str:
    audio_path = video_path.replace(".mp4", ".wav")
    ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000').overwrite_output().run(quiet=True)
    return audio_path

def extract_frames(video_path: str, mode: str = "uniform", max_frames: int = 5) -> list[str]:
    """
    Extracts frames from a video using either 'uniform' or 'scene' strategy.

    Args:
        video_path: Path to the .mp4 file
        mode: 'uniform' or 'scene'
        max_frames: Max number of frames to extract

    Returns:
        List of paths to extracted frame images
    """
    frame_dir = tempfile.mkdtemp()
    frame_pattern = os.path.join(frame_dir, "frame_%03d.jpg")

    if mode == "scene":
        (
            ffmpeg
            .input(video_path)
            .output(frame_pattern,
                    vf="select='gt(scene,0.3)',showinfo",
                    vsync="vfr")
            .overwrite_output()
            .run(quiet=True)
        )
    else:  # default to uniform
        (
            ffmpeg
            .input(video_path)
            .output(frame_pattern,
                    vf="select=not(mod(n\\,30))",
                    vframes=max_frames)
            .overwrite_output()
            .run(quiet=True)
        )

    all_frames = sorted(os.listdir(frame_dir))[:max_frames]
    return [os.path.join(frame_dir, f) for f in all_frames]

def get_clip_vector(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt")
    return clip_model.get_image_features(**inputs).detach().numpy()[0]

def average_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(vectors), axis=0)

def fuse_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2


### === ENDPOINTS === ###

class Query(BaseModel):
    text: str

@app.post("/embed/text")
def embed_text(q: Query):
    vector = chunk_and_embed(q.text)
    return {"embedding": vector.tolist()}


@app.post("/embed/video")
async def embed_video(
    file: UploadFile = File(...),
    caption: str = Form(""),
    tags: str = Form(""),
    frame_mode: str = Form("uniform")  # or "scene"
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    # Transcribe audio
    audio_path = extract_audio(tmp_path)
    transcript = asr_model.transcribe(audio_path)["text"]
    os.remove(audio_path)

    # Extract and encode multiple frames
    frame_paths = extract_frames(tmp_path, mode=frame_mode, max_frames=5)
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    features = clip_model.get_image_features(**inputs).detach().numpy()
    clip_vector = np.mean(features, axis=0)

    # Cleanup
    for p in frame_paths:
        os.remove(p)
    os.remove(tmp_path)

    # Encode text and fuse
    combined_text = f"{caption} {tags} {transcript}".strip()
    text_vector = chunk_and_embed(combined_text)

    alpha = 0.7
    beta = 0.3
    fused_vector = (alpha * text_vector + beta * clip_vector).tolist()

    return {
        "embedding": fused_vector,
        "transcript": transcript,
        "frame_mode": frame_mode
    }


@app.post("/embed/images")
async def embed_images(
    files: list[UploadFile] = File(...),
    caption: str = Form(""),
    tags: str = Form("")
):
    images = [Image.open(await f.read()).convert("RGB") for f in files]
    image_vectors = [get_clip_vector(img) for img in images]
    image_vector = average_vectors(image_vectors)

    if caption or tags:
        text_vector = chunk_and_embed(f"{caption} {tags}")
        final_vector = fuse_vectors(text_vector, image_vector)
    else:
        final_vector = image_vector

    return {"embedding": final_vector.tolist()}

# embed-server

A modular, production-ready API server for generating text, video, and preference embeddings using state-of-the-art transformer models. Built with FastAPI, this server is designed for scalable integration in modern ML and data workflows.

---

## Features

- **Text Embedding:** Generate sentence-transformer embeddings with automatic chunking for long inputs.
- **Video Embedding:** Combines CLIP visual embeddings with audio transcription (OpenAI Whisper) and rich metadata (captions, tags).
- **Preference Aggregation:** Weighted averaging and projection of heterogeneous embeddings (text/video) for user or profile modeling.
- **Efficient Model Management:** All models are loaded once and cached for efficiency.
- **Production Quality:** Robust error handling, modular codebase, Docker support, and clear API schema.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
  - [/embed/text](#embedtext)
  - [/embed/video](#embedvideo)
  - [/embed/aggregate](#embedaggregate)
- [Usage Examples](#usage-examples)
- [Architecture Overview](#architecture-overview)
- [Notes & Requirements](#notes--requirements)
- [License](#license)

---

## Installation

### Prerequisites

- Python 3.8+
- `ffmpeg` (must be installed and available in your `PATH`)
- (For video endpoints) Sufficient system memory and CPU
- (Recommended) Docker for production deployments

### Installing with pip (development/local)

```bash
git clone https://github.com/Konsultn-Engineering/embed-server.git
cd embed-server
pip install -r requirements.txt
```

### Running with Docker (production)

Build and run the server in a container:

```bash
docker build -t embed-server .
docker run -p 8000:8000 embed-server
```

---

## Running the Server

### Development

```bash
uvicorn embed_server:app --host 0.0.0.0 --port 8000 --reload
```

### Production (as in Dockerfile)

```bash
uvicorn embed_server:app --host 0.0.0.0 --port 8000
```

---

## API Reference

All endpoints are served from the root (no version prefix). Content type is JSON unless uploading files.

---

### `/embed/text`

**POST**  
Generate a text embedding.

#### Request body

```json
{
  "text": "Some text to embed"
}
```

#### Response

```json
{
  "embedding": [0.123, 0.456, ...]
}
```

---

### `/embed/video`

**POST**  
Generate video, content, and transcript embeddings from an uploaded video.

#### Form Data

- `file`: Video file (`.mp4`)
- `caption`: (optional) Caption string
- `tags`: (optional) Tags string
- `frame_mode`: (optional, default: `uniform`) Frame extraction mode

#### Response

```json
{
  "video_embedding": [...],
  "content_embedding": [...],
  "transcript_embedding": [...],
  "transcript": "Full transcript"
}
```

---

### `/embed/aggregate`

**POST**  
Weighted aggregation of multiple embeddings (supports text and video vectors).

#### Request body

```json
{
  "vectors": [
    {"embedding": [0.1, 0.2, ...], "weight": 0.8},
    {"embedding": [0.3, 0.4, ...], "weight": 0.2}
  ]
}
```

#### Response

```json
{
  "embedding": [0.123, 0.456, ...]
}
```

---

## Usage Examples

### Text Embedding

```bash
curl -X POST http://localhost:8000/embed/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!"}'
```

### Video Embedding

```bash
curl -X POST http://localhost:8000/embed/video \
  -F "file=@your_video.mp4" \
  -F "caption=My Video" \
  -F "tags=tag1,tag2"
```

### Preference/Embedding Aggregation

```bash
curl -X POST http://localhost:8000/embed/aggregate \
  -H "Content-Type: application/json" \
  -d '{"vectors":[{"embedding":[0.1,0.2,...],"weight":1.0},{"embedding":[0.3,0.4,...],"weight":0.5}]}'
```

---

## Architecture Overview

- **Entrypoint:** `embed_server.py` (FastAPI app)
- **Routes:** Defined in `routes.py`
- **Models:** Pydantic schemas in `models.py`
- **Embeddings:** Model loading and embedding logic in `embedding.py`
- **Video utilities:** (not shown) used by video embedding endpoint

Models loaded:

- Text: `all-MiniLM-L6-v2` (sentence-transformers)
- Visual: `openai/clip-vit-base-patch32` (CLIP)
- Audio: OpenAI Whisper `base` (for ASR)

---

## Notes & Requirements

- The `/embed/video` endpoint requires `ffmpeg` and can be resource-intensive.
- All endpoints return a 500 error for unexpected failures.
- Only `.mp4` video files are currently supported.
- Vector dimensions:
  - Text: 384
  - Visual: 512
- Video and text vectors are projected/aggregated for preference modeling.
- Some randomness exists in visual-to-text projection (replace with learned mapping for production ML use).

---

## License

&copy; Konsultn Engineering. All rights reserved.

---

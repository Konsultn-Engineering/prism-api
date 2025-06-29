# embed-server

A modular, production-ready API server for generating text, video, and preference embeddings using state-of-the-art transformer models. Built with FastAPI, this server is designed for scalable integration into modern ML and search pipelines.

## Features

- **Text Embedding:** Sentence-transformer based, with automatic chunking for long inputs.
- **Video Embedding:** Combines CLIP visual embeddings with audio transcription (Whisper) and rich metadata (captions, tags).
- **Preference Aggregation:** Weighted averaging and projection of heterogeneous embeddings for user/profile modeling.
- **Efficient Model Management:** All models are loaded once and cached via `lru_cache`.
- **Production Quality:** Robust error handling, modular codebase, and clear API schema.

## Quick Start

### Prerequisites

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/download.html) installed and available in your `PATH`

### Installation

````bash
git clone https://github.com/Konsultn-Engineering/embed-server.git
cd embed-server
pip install -r requirements.txt

### Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

````

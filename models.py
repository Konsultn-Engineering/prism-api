from pydantic import BaseModel

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
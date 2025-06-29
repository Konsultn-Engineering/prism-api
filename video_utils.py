"""
video_utils.py

Utility functions for video and audio processing in the embed-server project.

This module provides methods for extracting audio from video files and extracting frames
(either uniformly or by scene change) using ffmpeg. All temporary files are managed using
Python's tempfile and os libraries.
"""

import tempfile
import os
import ffmpeg

def extract_audio(video_path: str) -> str:
    """
    Extracts the audio track from a video file and saves it as a WAV file.

    Args:
        video_path (str): Path to the input video file (e.g., .mp4).

    Returns:
        str: Path to the extracted WAV audio file.

    Raises:
        ffmpeg.Error: If ffmpeg fails to extract audio.
    """
    audio_path = video_path.replace(".mp4", ".wav")
    ffmpeg.input(video_path).output(
        audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000'
    ).overwrite_output().run(quiet=True)
    return audio_path

def extract_frames(video_path: str, mode: str = "uniform", max_frames: int = 5) -> list[str]:
    """
    Extracts frames from a video file, either uniformly or at scene changes.

    Args:
        video_path (str): Path to the input video file.
        mode (str, optional): Extraction mode: "uniform" for evenly-spaced frames,
            "scene" for scene-change-based frames. Defaults to "uniform".
        max_frames (int, optional): Maximum number of frames to extract. Defaults to 5.

    Returns:
        list[str]: List of paths to the extracted frame image files (JPEG).

    Raises:
        ffmpeg.Error: If ffmpeg fails to extract frames.
    """
    frame_dir = tempfile.mkdtemp()
    pattern = os.path.join(frame_dir, "frame_%03d.jpg")
    if mode == "scene":
        vf_expr = "select='gt(scene,0.3)',showinfo"
        vsync = "vfr"
    else:
        vf_expr = f"select=not(mod(n\\,{30}))"
        vsync = None
    output_kwargs = dict(vf=vf_expr, vframes=max_frames)
    if vsync:
        output_kwargs["vsync"] = vsync
    ffmpeg.input(video_path).output(
        pattern,
        **output_kwargs
    ).overwrite_output().run(quiet=True)
    frames = sorted(os.listdir(frame_dir))[:max_frames]
    return [os.path.join(frame_dir, f) for f in frames]
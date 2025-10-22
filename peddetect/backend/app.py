"""FastAPI application for HOG-based pedestrian detection."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import cv2
import imageio.v2 as imageio
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
MAX_REPORTED_FRAMES = 50
TRACK_DISTANCE_THRESHOLD = 80.0
TRACK_MAX_MISSES = 10
SLOW_SPEED_THRESHOLD = 60.0
FAST_SPEED_THRESHOLD = 150.0
app = FastAPI()
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]


def _load_allowed_origins() -> List[str]:
    raw = os.getenv("ALLOWED_ORIGINS")
    if not raw:
        return DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Pedestrian Detection API"}


ALLOWED_ORIGINS = _load_allowed_origins()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Pedestrian Detection Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")


def _detect_and_annotate(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """Process a video with HOG-based pedestrian detection and annotate frames."""

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("Unable to open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width == 0 or height == 0:
        cap.release()
        raise ValueError("Video metadata could not be read; ensure the file is a valid video.")

    try:
        video_writer = imageio.get_writer(
            str(output_path),
            fps=max(float(fps), 1.0),
            codec="libx264",
            format="FFMPEG",
            bitrate="6000k",
            macro_block_size=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"],
        )
    except Exception as exc:  # pragma: no cover - FFmpeg configuration issues
        cap.release()
        raise ValueError("Unable to initialise the video encoder; ensure FFmpeg is available.") from exc

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    total_frames = 0
    frames_with_detections = 0
    total_detections = 0
    reported_frames: List[Dict[str, Any]] = []

    tracks: List[Dict[str, Any]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes, weights = hog.detectMultiScale(
                frame,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )

            frame_detections: List[Dict[str, int]] = []
            centers: List[np.ndarray] = []
            for (x, y, w, h) in boxes:
                center = np.array([float(x + w / 2), float(y + h / 2)])
                centers.append(center)
                frame_detections.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

            if frame_detections:
                frames_with_detections += 1
                total_detections += len(frame_detections)
                if len(reported_frames) < MAX_REPORTED_FRAMES:
                    reported_frames.append(
                        {
                            "frame_index": total_frames,
                            "count": len(frame_detections),
                            "boxes": frame_detections,
                        }
                    )

            if centers:
                updated_tracks: List[Dict[str, Any]] = []
                unmatched_centers = list(zip(frame_detections, centers))

                for track in tracks:
                    best_match = None
                    best_distance = float("inf")
                    for idx, (det, center) in enumerate(unmatched_centers):
                        distance = np.linalg.norm(center - track["position"])
                        if distance < best_distance:
                            best_distance = distance
                            best_match = (idx, det, center)

                    if best_match and best_distance <= TRACK_DISTANCE_THRESHOLD:
                        idx, det, center = best_match
                        displacement = np.linalg.norm(center - track["position"])
                        track["position"] = center
                        track["velocity_samples"].append(displacement)
                        track["misses"] = 0
                        track["last_detection"] = det
                        unmatched_centers.pop(idx)
                        updated_tracks.append(track)
                    else:
                        track["misses"] += 1
                        if track["misses"] <= TRACK_MAX_MISSES:
                            updated_tracks.append(track)

                for det, center in unmatched_centers:
                    updated_tracks.append(
                        {
                            "position": center,
                            "velocity_samples": [],
                            "misses": 0,
                            "last_detection": det,
                        }
                    )

                tracks = updated_tracks

            # Draw boxes and movement status
            for track in tracks:
                det = track.get("last_detection")
                if not det:
                    continue
                x = det["x"]
                y = det["y"]
                w = det["w"]
                h = det["h"]

                avg_speed = 0.0
                if track["velocity_samples"]:
                    avg_speed = float(np.mean(track["velocity_samples"]))

                if avg_speed >= FAST_SPEED_THRESHOLD:
                    label = "fast"
                elif avg_speed <= SLOW_SPEED_THRESHOLD:
                    label = "slow"
                else:
                    label = "normal"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                text = f"speed: {label}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                text_x = x + w - tw - 6
                text_y = max(y + 20, th + 10)

                cv2.rectangle(frame, (text_x - 4, text_y - th - 6), (text_x + tw + 4, text_y + 4), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            if video_writer is not None:
                video_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            total_frames += 1
    finally:
        cap.release()
        if 'video_writer' in locals():
            video_writer.close()

    if total_frames == 0:
        raise ValueError("No frames detected; the uploaded file may be corrupted.")

    return {
        "total_frames": total_frames,
        "frames_with_detections": frames_with_detections,
        "total_detections": total_detections,
        "reported_frames": reported_frames,
    }


@app.post("/process-video")
async def process_video(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a video upload, run pedestrian detection, and return summary metadata."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="A file must be provided.")

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="Only video uploads are supported.")

    video_id = uuid4().hex
    suffix = Path(file.filename).suffix or ".mp4"
    input_path = UPLOAD_DIR / f"{video_id}{suffix}"
    output_path = PROCESSED_DIR / f"{video_id}.mp4"

    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        summary = _detect_and_annotate(input_path, output_path)
    except ValueError as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected runtime errors
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from exc
    finally:
        file.file.close()
        if input_path.exists():
            input_path.unlink(missing_ok=True)

    payload: Dict[str, Any] = {
        "message": "Video processed successfully.",
        "processed_video_url": f"/processed/{output_path.name}",
        **summary,
    }

    return JSONResponse(content=payload)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Simple health endpoint for readiness checks."""

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# Pedestrian Detection with HOG

This project combines a Next.js 14 front end with a FastAPI backend to perform pedestrian detection on uploaded videos using the Histogram of Oriented Gradients (HOG) descriptor. Users can upload a clip, trigger server-side processing via OpenCV, and review annotated playback alongside summary statistics.

## Project Structure

- `src/` – Next.js App Router UI for uploading videos and visualizing results.
- `backend/` – FastAPI service that runs pedestrian detection and serves processed videos.
- `public/` – Static assets served by Next.js.

## Prerequisites

- Node.js 18 or later
- npm 9 or later
- Python 3.10 or later with pip
- OpenCV runtime dependencies (OS packages required by `opencv-python`)

## Setup

1. **Install front-end dependencies**

	```bash
	cd pedestrian-hog
	npm install
	```

2. **Install backend dependencies**

	```bash
	cd backend
	python -m venv .venv
	.venv\Scripts\activate
	pip install -r requirements.txt
	```

3. **Configure environment variables**

	Create a `.env.local` file in the project root and set the API base URL (defaults to `http://localhost:8000` if omitted):

	```bash
	NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
	```

## Running the Stack Locally

Open two terminals:

1. **FastAPI backend**

	```bash
	cd pedestrian-hog/backend
	.venv\Scripts\activate
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload
	```

	The backend exposes `POST /process-video` for analysis, `GET /processed/*` for annotated videos, and `GET /health` for readiness checks.

2. **Next.js front end**

	```bash
	cd pedestrian-hog
	npm run dev
	```

	Visit `http://localhost:3000` to access the web interface.

## Pedestrian Detection Workflow

1. The browser uploads the selected video to `POST /process-video`.
2. FastAPI stores the upload temporarily, uses OpenCV's default people detector with HOG + SVM to scan each frame, and draws bounding boxes around detected pedestrians.
3. Processed frames are written to a new MP4 in `backend/processed/` and served via static hosting.
4. The API responds with aggregate statistics and the relative URL of the annotated video, which the front end renders in a player alongside detection summaries.

## API Reference

### `POST /process-video`

- **Request**: `multipart/form-data` with a `file` field containing a video.
- **Response** (`200`):

  ```json
  {
	 "message": "Video processed successfully.",
	 "processed_video_url": "/processed/<generated>.mp4",
	 "total_frames": 123,
	 "frames_with_detections": 45,
	 "total_detections": 62,
	 "reported_frames": [
		{ "frame_index": 10, "count": 2, "boxes": [{ "x": 100, "y": 120, "w": 48, "h": 96 }] }
	 ]
  }
  ```

  The backend caps `reported_frames` at 50 entries to keep responses lightweight.

### `GET /processed/<filename>.mp4`

Static endpoint serving annotated videos for streaming or download.

### `GET /health`

Health probe returning `{ "status": "ok" }` when the service is ready.

## Notes

- Large videos can take significant time to process; keep uploads to short clips for faster feedback.
- Uploaded source files are deleted after processing, while annotated outputs remain in `backend/processed/` until manually cleared.
- Replace sample videos with your own footage by uploading through the UI; support for batch processing can be added later if needed.

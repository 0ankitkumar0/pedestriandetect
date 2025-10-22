"use client";

import { FormEvent, useMemo, useState } from "react";

type FrameSummary = {
  frame_index: number;
  count: number;
  boxes: Array<{ x: number; y: number; w: number; h: number }>;
};

type ProcessVideoResponse = {
  message: string;
  processed_video_url?: string;
  total_frames: number;
  frames_with_detections: number;
  total_detections: number;
  reported_frames: FrameSummary[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ProcessVideoResponse | null>(null);

  const processedVideoHref = useMemo(() => {
    if (!result?.processed_video_url) {
      return null;
    }

    try {
      return new URL(result.processed_video_url, API_BASE_URL).toString();
    } catch {
      return `${API_BASE_URL.replace(/\/$/, "")}${result.processed_video_url}`;
    }
  }, [result?.processed_video_url]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Choose a video file before uploading.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    setIsSubmitting(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/process-video`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Processing failed.");
      }

      const payload = (await response.json()) as ProcessVideoResponse;
      setResult(payload);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred. Try again.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <main className="mx-auto flex max-w-4xl flex-col gap-8 px-6 py-16">
        <header className="space-y-3 text-center sm:text-left">
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
            Pedestrian Detection with HOG
          </h1>
          <p className="text-slate-300">
            Upload a short video clip to detect pedestrian activity using the Histogram of Oriented Gradients (HOG) detector.
          </p>
        </header>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg">
          <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
            <label className="flex w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-700 bg-slate-900/40 px-6 py-10 text-center transition hover:border-slate-500">
              <span className="text-lg font-medium">Select a video file</span>
              <span className="text-sm text-slate-400">
                MP4, AVI, MOV or other common video formats supported by OpenCV.
              </span>
              <input
                accept="video/*"
                className="hidden"
                id="video"
                name="video"
                type="file"
                onChange={(event) => {
                  const file = event.target.files?.[0] ?? null;
                  setSelectedFile(file);
                }}
              />
              {selectedFile ? (
                <span className="rounded-full bg-slate-800 px-4 py-1 text-sm text-slate-200">
                  {selectedFile.name}
                </span>
              ) : (
                <span className="rounded-full bg-slate-800 px-4 py-1 text-sm text-slate-400">
                  No file chosen
                </span>
              )}
            </label>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <button
                type="submit"
                className="inline-flex items-center justify-center rounded-lg bg-emerald-500 px-5 py-2 text-sm font-semibold text-emerald-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={isSubmitting}
              >
                {isSubmitting ? "Processing..." : "Upload & Analyze"}
              </button>
              <p className="text-xs text-slate-500">
                Processing happens locally on this demo server; large files may take a while.
              </p>
            </div>
          </form>

          {error && (
            <p className="mt-4 rounded-lg border border-rose-500/50 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              {error}
            </p>
          )}
        </section>

        {result && (
          <section className="space-y-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg">
            <header className="space-y-1">
              <h2 className="text-2xl font-semibold text-slate-50">Analysis Summary</h2>
              <p className="text-sm text-slate-400">{result.message}</p>
            </header>

            <dl className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-3">
                <dt className="text-xs uppercase tracking-wide text-slate-500">Frames Processed</dt>
                <dd className="text-xl font-semibold text-slate-100">{result.total_frames}</dd>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-3">
                <dt className="text-xs uppercase tracking-wide text-slate-500">Frames With Detections</dt>
                <dd className="text-xl font-semibold text-slate-100">{result.frames_with_detections}</dd>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-3">
                <dt className="text-xs uppercase tracking-wide text-slate-500">Total Pedestrians Detected</dt>
                <dd className="text-xl font-semibold text-slate-100">{result.total_detections}</dd>
              </div>
            </dl>

            {processedVideoHref && (
              <div className="space-y-2">
                <h3 className="text-lg font-medium text-slate-100">Annotated Playback</h3>
                <video
                  controls
                  className="w-full overflow-hidden rounded-xl border border-slate-800"
                  src={processedVideoHref}
                >
                  Your browser does not support the video tag.
                </video>
              </div>
            )}

            {result.reported_frames.length > 0 && (
              <div className="space-y-2">
                <h3 className="text-lg font-medium text-slate-100">Sample Detection Frames</h3>
                <div className="max-h-60 overflow-y-auto rounded-xl border border-slate-800 bg-slate-900">
                  <table className="w-full text-left text-sm">
                    <thead className="sticky top-0 bg-slate-900/95 text-xs uppercase tracking-wide text-slate-400">
                      <tr>
                        <th className="px-4 py-2">Frame</th>
                        <th className="px-4 py-2">Detections</th>
                        <th className="px-4 py-2">Bounding Boxes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.reported_frames.map((frame) => (
                        <tr key={frame.frame_index} className="border-t border-slate-800 text-slate-200">
                          <td className="px-4 py-2">{frame.frame_index}</td>
                          <td className="px-4 py-2">{frame.count}</td>
                          <td className="px-4 py-2 text-xs">
                            {frame.boxes
                              .map((box) => `(${box.x}, ${box.y}, ${box.w}, ${box.h})`)
                              .join(", ")}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-xs text-slate-500">
                  Showing up to {Math.min(result.reported_frames.length, 50)} frames with pedestrian detections.
                </p>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

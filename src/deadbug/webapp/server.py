"""FastAPI server for the Dead Bug coach.

Three entry points, one engine:

    POST /api/upload      a file from the user's device
    POST /api/youtube     a URL (needs yt-dlp)
    WS   /ws/camera       live frames from the browser

**The camera runs server-side, not in the browser.** The browser captures with
``getUserMedia``, sends JPEG frames over the socket, and receives landmarks and
session state back. Running MediaPipe in JavaScript instead would be smoother,
and it would also mean the demo is a *different implementation* from the one the
numbers came from -- at which point the demo stops being evidence about the
system and becomes evidence about a reimplementation of it. The round trip costs
latency; it buys the claim.

File and YouTube analyses run in a worker thread and report progress by
polling, rather than blocking the request. MediaPipe heavy runs at roughly
12 fps here, so a two-minute clip is several minutes of work.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import REPO_ROOT, cfg_get, load_config
from ..ingest.video_source import snap
from ..live.engine import CoachEngine
from .analysis import analyse_video

STATIC = Path(__file__).parent / "static"
WORK = REPO_ROOT / "data" / "webapp"
UPLOADS = WORK / "uploads"
RESULTS = WORK / "results"

#: Refuse anything longer. Extraction is linear in duration and the browser
#: would sit on a spinner for an hour.
MAX_UPLOAD_SECONDS = 600.0
MAX_UPLOAD_BYTES = 300 * 1024 * 1024

#: Longest side the camera frames are processed at. Bigger is not better here:
#: MediaPipe is the bottleneck and a slower loop means fewer samples per second
#: of movement.
CAMERA_WIDTH = 640


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"        # queued | running | done | error
    progress: float = 0.0
    message: str = ""
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "status": self.status,
            "progress": round(self.progress, 3), "message": self.message,
            "result": self.result, "error": self.error,
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


def create_app(config_path: str = "configs/base.yaml") -> FastAPI:
    cfg = load_config(config_path)
    app = FastAPI(title="Dead Bug Coach", docs_url="/api/docs")
    jobs = JobStore()
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="deadbug")

    for directory in (UPLOADS, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)

    app.state.cfg = cfg
    app.state.jobs = jobs

    # ------------------------------------------------------------------
    # pages
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "model": cfg_get(cfg, f"pose.model_paths.{cfg_get(cfg, 'pose.variant')}"),
            "youtube": _has_ytdlp(),
            "max_seconds": MAX_UPLOAD_SECONDS,
        }

    # ------------------------------------------------------------------
    # analysis jobs
    # ------------------------------------------------------------------

    def _run(job: Job, path: Path, setup_seconds: float, baseline_reps: int) -> None:
        job.status = "running"
        try:
            annotated = RESULTS / f"{job.id}.mp4"
            result = analyse_video(
                path, cfg,
                setup_seconds=setup_seconds, baseline_reps=baseline_reps,
                progress=lambda frac, msg: _update(job, frac, msg),
                annotate_to=annotated,
                max_seconds=MAX_UPLOAD_SECONDS,
            )
            result["job_id"] = job.id
            result["annotated_url"] = f"/api/result/{job.id}/video" if annotated.exists() else None
            (RESULTS / f"{job.id}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            job.result = result
            job.status = "done"
            job.progress = 1.0
            job.message = "done"
        except Exception as exc:                     # noqa: BLE001 - surfaced to the user
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "failed"

    def _update(job: Job, frac: float, msg: str) -> None:
        job.progress = frac
        job.message = msg

    @app.post("/api/upload")
    async def upload(
        file: UploadFile = File(...),
        setup_seconds: float = Form(3.0),
        baseline_reps: int = Form(3),
    ) -> JSONResponse:
        suffix = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
            raise HTTPException(400, f"unsupported file type: {suffix}")

        job = jobs.create("upload")
        target = UPLOADS / f"{job.id}{suffix}"
        size = 0
        with open(target, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(413, "file larger than 300 MB")
                out.write(chunk)

        ok, why = _probe_ok(target)
        if not ok:
            target.unlink(missing_ok=True)
            raise HTTPException(400, why)

        pool.submit(_run, job, target, setup_seconds, baseline_reps)
        return JSONResponse(job.as_dict())

    @app.post("/api/youtube")
    async def youtube(
        url: str = Form(...),
        setup_seconds: float = Form(3.0),
        baseline_reps: int = Form(3),
    ) -> JSONResponse:
        if not _has_ytdlp():
            raise HTTPException(
                501,
                "yt-dlp is not installed. It is an optional extra: "
                "pip install yt-dlp",
            )
        from ..ingest.download import download, probe

        job = jobs.create("youtube")
        try:
            info = await asyncio.to_thread(probe, url)
        except Exception as exc:                     # noqa: BLE001
            raise HTTPException(400, f"could not read that URL: {exc}") from exc

        duration = info.get("duration_s") or 0
        if duration and duration > MAX_UPLOAD_SECONDS:
            raise HTTPException(
                400,
                f"that video is {duration/60:.0f} minutes. The limit is "
                f"{MAX_UPLOAD_SECONDS/60:.0f} - extraction cost is linear in "
                "duration, and instructional videos are mostly not exercise.",
            )

        def fetch_and_run() -> None:
            job.status = "running"
            job.message = "downloading"
            try:
                path = download(
                    url, UPLOADS,
                    max_height=cfg_get(cfg, "ingest.download.max_height"),
                    max_duration_s=MAX_UPLOAD_SECONDS,
                )
            except Exception as exc:                 # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                return
            _run(job, Path(path), setup_seconds, baseline_reps)
            if job.result is not None:
                job.result["title"] = info.get("title")

        pool.submit(fetch_and_run)
        return JSONResponse({**job.as_dict(), "title": info.get("title"),
                             "duration_s": duration})

    @app.get("/api/job/{job_id}")
    def job_status(job_id: str) -> JSONResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return JSONResponse(job.as_dict())

    @app.get("/api/result/{job_id}/video")
    def job_video(job_id: str) -> FileResponse:
        path = RESULTS / f"{job_id}.mp4"
        if not path.exists():
            raise HTTPException(404, "no annotated video for that job")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/result/{job_id}/json")
    def job_json(job_id: str) -> FileResponse:
        path = RESULTS / f"{job_id}.json"
        if not path.exists():
            raise HTTPException(404, "no result for that job")
        return FileResponse(path, media_type="application/json",
                            filename=f"deadbug_{job_id}.json")

    # ------------------------------------------------------------------
    # live camera
    # ------------------------------------------------------------------

    @app.websocket("/ws/camera")
    async def camera(ws: WebSocket) -> None:
        await ws.accept()
        engine: CoachEngine | None = None
        frame_index = 0
        t0 = time.monotonic()

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                kind = msg.get("type")
                if kind == "stop":
                    break
                if kind == "reset":
                    if engine is not None:
                        engine.close()
                    engine = None
                    frame_index = 0
                    t0 = time.monotonic()
                    await ws.send_text(json.dumps({"type": "reset"}))
                    continue
                if kind != "frame":
                    continue

                frame = _decode(msg.get("data", ""))
                if frame is None:
                    continue

                if engine is None:
                    snap_to = cfg_get(cfg, "ingest.snap_dims_to")
                    size = (snap(frame.shape[1], snap_to), snap(frame.shape[0], snap_to))
                    engine = CoachEngine(
                        cfg, fps=float(msg.get("fps") or 15.0), frame_size=size,
                        setup_seconds=float(msg.get("setup_seconds") or 3.0),
                        baseline_reps=int(msg.get("baseline_reps") or 3),
                    )
                    t0 = time.monotonic()
                    frame_index = 0

                size = engine.frame_size
                if (frame.shape[1], frame.shape[0]) != size:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

                # Wall clock: for a camera it IS the truth, and unlike a file
                # replay there is no media timeline to prefer.
                now = time.monotonic() - t0
                result = await asyncio.to_thread(engine.process, frame, frame_index, now)
                frame_index += 1
                await ws.send_text(json.dumps(_frame_payload(result)))

        except WebSocketDisconnect:
            pass
        finally:
            if engine is not None:
                engine.finish()
                payload = {"type": "report", "report": engine.report()}
                try:
                    await ws.send_text(json.dumps(_sanitise(payload)))
                except Exception:                    # noqa: BLE001 - socket already gone
                    pass
                engine.close()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        pool.shutdown(wait=False, cancel_futures=True)

    return app


# ----------------------------------------------------------------------


def _frame_payload(result) -> dict:
    state = result.state
    verdict = state.last_verdict
    payload = {
        "type": "frame",
        "frame": result.frame_index,
        "detected": result.detected,
        "phase": state.phase.value,
        "prompt": state.prompt,
        "progress": state.progress,
        "reps_done": state.reps_done,
        "reps_correct": state.reps_correct,
        "highlight": state.highlight,
        "lumbar_gap": result.lumbar_gap,
        "live_z": state.live_z,
        "floor_ready": result.floor_ready,
        "latency_ms": result.latency_ms,
        # Only the joints the overlay draws, as [x, y, visibility] in [0,1].
        "kpts": (
            [[float(p[0]), float(p[1]), float(p[3])] for p in result.kpts]
            if result.kpts is not None else None
        ),
        "verdict": (
            {"rep": verdict.rep_index, "side": verdict.side, "ok": bool(verdict.ok),
             "errors": list(verdict.errors), "message": verdict.message}
            if verdict is not None else None
        ),
    }
    return _sanitise(payload)


def _decode(data: str) -> np.ndarray | None:
    """Decode a browser data URL into BGR pixels."""
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    except (binascii.Error, ValueError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _probe_ok(path: Path) -> tuple[bool, str]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, "that file could not be opened as a video"
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    finally:
        cap.release()
    if frames <= 0:
        return False, "that file contains no readable frames"
    if fps > 0 and frames / fps > MAX_UPLOAD_SECONDS:
        return False, (
            f"that clip is {frames / fps / 60:.0f} minutes; the limit is "
            f"{MAX_UPLOAD_SECONDS / 60:.0f}"
        )
    return True, ""


def _has_ytdlp() -> bool:
    try:
        import yt_dlp  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def _sanitise(obj: Any) -> Any:
    """JSON has no NaN or numpy scalars."""
    import math

    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitise(obj.tolist())
    return obj


app = None


def main() -> int:
    """``python -m deadbug.webapp.server`` -- convenience launcher."""
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Run the Dead Bug coach web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    print(f"\n  Dead Bug Coach  ->  http://{args.host}:{args.port}\n")
    uvicorn.run(create_app(args.config), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Writing a sub-clip to disk -- which the pipeline itself never needs.

Read this before reaching for it. **Trimming is not part of the pipeline.**
``configs/base.yaml`` sets ``ingest.clip.enabled: false`` and the reason is in
:mod:`deadbug.ingest.video_source`: a sub-clip is a *seek*, not a new file.
``VideoSource(path, start_s=..., end_s=...)`` iterates exactly the requested
range, and :func:`deadbug.pose.mediapipe_backbone.extract_clip` takes the same
two arguments. Nothing downstream is made simpler by having a trimmed copy, and
producing one costs a re-encode that the whole fps policy exists to avoid.

So this module is for the cases outside the pipeline, where a *file* is the
deliverable: a five-second excerpt for a slide, or an isolated clip to hand to
someone else. It writes with OpenCV rather than ffmpeg, which keeps the
dependency list unchanged and means the output is re-encoded -- acceptable for a
presentation asset, not acceptable as pipeline input.

:func:`export_segments` is the one genuinely useful entry point: it takes the
segments :mod:`deadbug.segment.activity` found and writes each as its own file,
which turns "where in this 98-second video is the exercise" into something you
can actually watch.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from .video_source import VideoSource

#: OpenCV writes mp4 through this on Windows without extra codecs installed.
FOURCC = "mp4v"


def write_subclip(
    src: str | Path,
    dst: str | Path,
    start_s: float | None = None,
    end_s: float | None = None,
    snap_to: int = 16,
    fourcc: str = FOURCC,
) -> Path:
    """Write ``[start_s, end_s)`` of ``src`` to ``dst``. Returns ``dst``.

    **Re-encodes.** Do not feed the result back into the pipeline -- extract from
    the original with ``start_s`` / ``end_s`` instead, which is both lossless and
    faster.
    """
    source = VideoSource(src, snap_to=snap_to, start_s=start_s, end_s=end_s)
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*fourcc),
        source.meta.fps,
        (source.meta.out_width, source.meta.out_height),
    )
    if not writer.isOpened():
        raise OSError(f"cannot open a writer for {out} with fourcc {fourcc!r}")
    try:
        n_written = 0
        for _index, frame in source:
            writer.write(frame)
            n_written += 1
    finally:
        writer.release()

    if n_written == 0:
        out.unlink(missing_ok=True)
        raise ValueError(
            f"no frames in [{start_s}, {end_s}) of {src} -- nothing written"
        )
    return out


def export_segments(
    src: str | Path,
    segments,
    out_dir: str | Path,
    stem: str | None = None,
    snap_to: int = 16,
) -> list[Path]:
    """Write one file per detected exercise segment.

    ``segments`` is the list from
    :func:`deadbug.segment.activity.find_segments`. Files are named
    ``<stem>_seg00_12.3-31.7s.mp4`` so the timings survive being copied into a
    slide deck.
    """
    src = Path(src)
    stem = stem or src.stem
    out_dir = Path(out_dir)

    written = []
    for i, segment in enumerate(segments):
        name = f"{stem}_seg{i:02d}_{segment.start_s:.1f}-{segment.end_s:.1f}s.mp4"
        written.append(
            write_subclip(src, out_dir / name, segment.start_s, segment.end_s, snap_to)
        )
    return written

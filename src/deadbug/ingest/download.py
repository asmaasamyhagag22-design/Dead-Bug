"""Fetch a source video by URL, without re-encoding it.

The app's stated priority is that it runs on any YouTube video the user picks,
and everything downstream already supports that -- ``run_live.py --source`` takes
a path, :class:`~deadbug.ingest.video_source.VideoSource` carries the true fps,
and :mod:`deadbug.segment.activity` finds the exercise inside a clip that is
mostly talking. The only missing link was getting the file onto disk.

Three constraints this module keeps, all of them inherited rather than invented:

**No re-encode.** ``requirements.txt`` deliberately excludes ffmpeg, and the
whole fps policy exists so that clips can arrive at 23.98 / 25 / 29.97 / 30 and
be handled as they are. So the format selector asks for a *progressive* mp4 --
one file with audio and video already muxed -- rather than the separate streams
yt-dlp would otherwise have to merge with ffmpeg. On the rare video with no
progressive rendition this fails loudly instead of silently transcoding.

**yt-dlp is optional.** It is not in ``requirements.txt``. Every function here
raises a message naming the install command rather than an ``ImportError``, so a
machine that only ever runs the app on local files never needs it.

**The measured expectation, so nobody is surprised:** a YouTube instructional
video is a source of *coaching*, not a source of reps. ``videoplayback (3)`` is
98 seconds and contains zero reps -- confirmed independently by the live counter
and the offline segmenter. :func:`download` therefore returns the path and lets
the caller run activity detection; it makes no promise that the clip is usable.
"""

from __future__ import annotations

import re
from pathlib import Path

#: One muxed file, no merge step, capped at the configured height. The trailing
#: fallbacks are ordered so that ffmpeg is never required.
FORMAT_TEMPLATE = "best[ext=mp4][height<={height}]/best[ext=mp4]/best"

_YTDLP_HINT = (
    "yt-dlp is not installed. It is an optional extra, kept out of "
    "requirements.txt on purpose:\n"
    "    ./venv/Scripts/python.exe -m pip install yt-dlp"
)


def _require_ytdlp():
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(_YTDLP_HINT) from exc
    return yt_dlp


def safe_stem(text: str, max_len: int = 60) -> str:
    """A filesystem-safe stem. Windows rejects far more characters than POSIX."""
    cleaned = re.sub(r"[^\w\-. ]+", "_", text, flags=re.UNICODE).strip(" ._")
    return (cleaned[:max_len] or "clip").rstrip(" ._")


def probe(url: str) -> dict:
    """Metadata only -- no bytes fetched.

    Useful before committing to a download: a 40-minute physiotherapy lecture
    and a 30-second demonstration look identical as URLs.
    """
    yt_dlp = _require_ytdlp()
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "duration_s": info.get("duration"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "webpage_url": info.get("webpage_url", url),
    }


def download(
    url: str,
    out_dir: str | Path,
    max_height: int = 720,
    max_duration_s: float | None = 400.0,
    filename_template: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download one video and return the path to the file on disk.

    Args:
        max_duration_s: refuse anything longer. A 40-minute video is 40 minutes
            of pose extraction for, typically, no reps at all. Pass None to
            disable the guard.
        overwrite: by default an existing file with the same id is returned
            as-is, so re-running a command is free.

    Raises:
        RuntimeError: if yt-dlp is absent, the video is too long, or no
            progressive mp4 exists.
    """
    yt_dlp = _require_ytdlp()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe(url)
    duration = info.get("duration_s")
    if max_duration_s and duration and duration > max_duration_s:
        raise RuntimeError(
            f"{info['title']!r} is {duration:.0f}s, over the {max_duration_s:.0f}s limit. "
            "Trim it at the source or raise ingest.download.max_duration_s -- but note "
            "that extraction cost is linear in duration and instructional videos are "
            "mostly not exercise."
        )

    stem = (
        filename_template.replace("%(id)s", str(info["id"]))
        .replace("%(uploader)s", safe_stem(str(info.get("uploader") or "unknown")))
        .replace(".%(ext)s", "")
        if filename_template
        else f"{safe_stem(str(info.get('uploader') or 'yt'))}__{info['id']}"
    )
    target = out_dir / f"{stem}.mp4"
    if target.exists() and not overwrite:
        return target

    options = {
        "format": FORMAT_TEMPLATE.format(height=max_height),
        "outtmpl": str(out_dir / f"{stem}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Never invoke ffmpeg: a merge or a recode would change the fps the rest
        # of the pipeline is carefully carrying through untouched.
        "postprocessors": [],
        "overwrites": overwrite,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])

    if target.exists():
        return target
    # yt-dlp picked a different container despite the selector.
    candidates = sorted(out_dir.glob(f"{stem}.*"))
    if not candidates:
        raise RuntimeError(f"download reported success but produced no file for {url}")
    if candidates[0].suffix != ".mp4":
        raise RuntimeError(
            f"no progressive mp4 rendition for {url} (got {candidates[0].suffix}). "
            "Merging the separate streams would need ffmpeg and would re-encode."
        )
    return candidates[0]


def download_many(
    urls: list[str], out_dir: str | Path, **kwargs
) -> tuple[list[Path], list[tuple[str, str]]]:
    """Download a list of URLs. Returns ``(paths, failures)``.

    Failures are collected rather than raised: one dead link should not abandon
    a batch that has already spent bandwidth on the rest.
    """
    paths, failures = [], []
    for url in urls:
        try:
            paths.append(download(url, out_dir, **kwargs))
        except Exception as exc:  # noqa: BLE001 - the reason is reported, not swallowed
            failures.append((url, str(exc)))
    return paths, failures


def read_queries(path: str | Path) -> list[str]:
    """Read ``configs/queries.txt``: one URL or search term per line, ``#`` comments."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def download_from_config(cfg: dict, urls: list[str] | None = None) -> tuple[list[Path], list]:
    """Download using the ``ingest.download`` block.

    Honours ``ingest.download.enabled``. It defaults to false so that a full
    pipeline run never reaches out to the network by accident.
    """
    from ..config import cfg_get, resolve_path

    if not cfg_get(cfg, "ingest.download.enabled", False) and urls is None:
        raise RuntimeError(
            "ingest.download.enabled is false. Pass URLs explicitly, or set it in "
            "configs/base.yaml -- the default is off so a pipeline run never hits "
            "the network unasked."
        )
    if urls is None:
        urls = read_queries(
            resolve_path(cfg, "paths.raw").parent.parent
            / cfg_get(cfg, "ingest.download.queries_file")
        )
    return download_many(
        urls,
        resolve_path(cfg, "paths.raw"),
        max_height=cfg_get(cfg, "ingest.download.max_height"),
        max_duration_s=cfg_get(cfg, "ingest.download.max_duration_s"),
        filename_template=cfg_get(cfg, "ingest.download.filename_template", None),
    )

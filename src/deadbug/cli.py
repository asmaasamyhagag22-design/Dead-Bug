"""One entry point for every stage.

    deadbug fetch <url>          download a source video (yt-dlp, optional)
    deadbug dedup                which clips are the same footage
    deadbug clips-template       write the data/clips.csv skeleton
    deadbug scan <video>         is there any exercise in this file?
    deadbug extract              pose + mask for every clip in the manifest
    deadbug build                the reps table
    deadbug qc                   the QC report
    deadbug band                 the LOSO normative band
    deadbug coach --source ...   the live app
    deadbug try <url|path>       fetch, scan, then coach -- the demo path

The stages are separate commands rather than one pipeline because they fail for
completely different reasons and at completely different costs. ``extract`` is
minutes per clip and hits MediaPipe; ``build`` is seconds and hits nothing.
Bundling them would mean re-paying the expensive one to fix a mistake in the
cheap one.

``try`` is the exception, and it exists for one reason: the thing the project is
judged on is whether a person can point it at a video and see it work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import REPO_ROOT, cfg_get, get_logger, load_config, resolve_path, seed_everything


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/base.yaml")


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_fetch(args) -> int:
    from .ingest.download import download, probe

    cfg = load_config(args.config)
    if args.probe_only:
        _print(probe(args.url))
        return 0
    path = download(
        args.url,
        resolve_path(cfg, "paths.raw"),
        max_height=cfg_get(cfg, "ingest.download.max_height"),
        max_duration_s=None if args.any_length else cfg_get(cfg, "ingest.download.max_duration_s"),
        overwrite=args.force,
    )
    print(path)
    return 0


def cmd_dedup(args) -> int:
    from .ingest.dedup import build_dedup_groups, write_report

    cfg = load_config(args.config)
    clips = sorted(resolve_path(cfg, "paths.clips").glob("*.mp4"))
    if not clips:
        print("no clips found", file=sys.stderr)
        return 1
    result = build_dedup_groups(
        clips,
        sample_every_s=cfg_get(cfg, "ingest.dedup.sample_every_s"),
        threshold=cfg_get(cfg, "ingest.dedup.hamming_threshold"),
        match_frac=cfg_get(cfg, "ingest.dedup.match_frac_threshold"),
    )
    out = write_report(result, resolve_path(cfg, "paths.reports") / "dedup.csv")
    print(f"wrote {out}")
    return 0


def cmd_clips_template(args) -> int:
    from .dataset.build import clips_template

    cfg = load_config(args.config)
    target = resolve_path(cfg, "paths.clips_manifest")
    if target.exists() and target.stat().st_size > 0 and not args.force:
        print(f"{target} already has content; pass --force to overwrite", file=sys.stderr)
        return 1
    records = clips_template(
        resolve_path(cfg, "paths.clips"),
        resolve_path(cfg, "paths.reports") / "dedup.csv",
        out=target,
    )
    print(f"wrote {target} with {len(records)} rows")
    print("Fill in person_id, condition and view by hand -- none of the three can be")
    print("inferred from the pixels, and guessing person_id breaks the LOSO split.")
    return 0


def cmd_scan(args) -> int:
    """Answer the one question that decides whether a video is worth anything."""
    import numpy as np

    from .pose.mediapipe_backbone import extract_clip
    from .segment.activity import find_segments, summarise

    cfg = load_config(args.config)
    seed_everything(cfg_get(cfg, "seed"))
    video = Path(args.video)
    interim = resolve_path(cfg, "paths.interim")
    npz = interim / f"_scan_{video.stem}.npz"

    if args.force or not npz.exists():
        extract_clip(video, npz, cfg)
    with np.load(npz) as data:
        kpts = data["kpts_raw"].astype(np.float64)
        fps = float(data["fps"])
        frame_size = tuple(int(v) for v in data["frame_size"])

    segments = find_segments(kpts, fps, frame_size)
    summary = summarise(segments, kpts.shape[0] / fps)
    summary["video"] = str(video)
    summary["detection_rate"] = round(
        float(np.mean(np.isfinite(kpts[:, :, :2]).any(axis=(1, 2)))), 3
    )
    _print(summary)
    if not segments:
        print(
            "\nNo exercise segment found. For an instructional video that is the "
            "expected answer, not a bug: measured on videoplayback (3), 98 seconds "
            "of coaching contains zero reps.",
            file=sys.stderr,
        )
    return 0


def cmd_extract(args) -> int:
    from .dataset.build import extract_or_load
    from .dataset.schema import read_clips

    cfg = load_config(args.config)
    log = get_logger("extract", cfg)
    clips = read_clips(resolve_path(cfg, "paths.clips_manifest"))
    for clip in clips:
        video = REPO_ROOT / clip.file
        log.info("extracting %s", clip.clip_id)
        extract_or_load(video, clip.clip_id, cfg, force=args.force)
    print(f"extracted {len(clips)} clips")
    return 0


def cmd_build(args) -> int:
    from .dataset.build import build_dataset

    cfg = load_config(args.config)
    seed_everything(cfg_get(cfg, "seed"))
    summary = build_dataset(cfg, force=args.force)
    diagnostics = summary.pop("diagnostics")
    _print(summary)

    empty = [d["clip_id"] for d in diagnostics if not d.get("n_reps")]
    if empty:
        print(f"\n{len(empty)} clip(s) produced no reps: {', '.join(empty)}", file=sys.stderr)
    (resolve_path(cfg, "paths.reports") / "build_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str), encoding="utf-8"
    )
    return 0


def cmd_qc(args) -> int:
    from .qc.report import run

    cfg = load_config(args.config)
    path = resolve_path(cfg, "paths.reports") / "build_diagnostics.json"
    if not path.exists():
        print(f"{path} not found -- run `deadbug build` first", file=sys.stderr)
        return 1
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    _print(run(diagnostics, cfg))
    return 0


def cmd_band(args) -> int:
    from .dataset.build import read_reps
    from .dataset.normative import build_from_reps, save_band

    cfg = load_config(args.config)
    processed = resolve_path(cfg, "paths.processed")
    table = processed / "reps.parquet"
    if not table.exists():
        table = processed / "reps.csv"
    if not table.exists():
        print("no reps table -- run `deadbug build` first", file=sys.stderr)
        return 1

    reps = read_reps(table)
    bands = build_from_reps(reps, cfg)
    out = save_band(bands, resolve_path(cfg, "paths.band"))
    print(f"wrote {out} -- {len(bands)} leave-one-subject-out band(s)")
    for person, band in sorted(bands.items()):
        print(f"  held out {person}: fitted on {band.n_reps} reps from {band.n_persons} others")

    if not bands:
        # Say why. An empty band file is the correct output for the data we
        # have, and it is also exactly what a silent bug would produce.
        eligible = reps[
            reps["condition"].isin(cfg_get(cfg, "dataset.band.conditions"))
            & reps["view"].isin(cfg_get(cfg, "dataset.band.views"))
        ]
        n_persons = eligible["person_id"].nunique()
        print(
            f"\nNo band could be fitted, and that is the honest result, not an error.\n"
            f"  eligible reps:     {len(eligible)} of {len(reps)}\n"
            f"  eligible subjects: {n_persons}\n"
            f"Leave-one-subject-out needs at least 2 subjects: holding one out has to\n"
            f"leave someone to fit on. The filter is condition in "
            f"{cfg_get(cfg, 'dataset.band.conditions')} and view in "
            f"{cfg_get(cfg, 'dataset.band.views')}, and only side views qualify because\n"
            f"the lumbar gap does not project onto the silhouette from any other angle.\n"
            f"See FILMING.md.",
            file=sys.stderr,
        )
    return 0


def cmd_coach(args) -> int:
    """Delegate to the live app, which owns its own argument parsing."""
    script = REPO_ROOT / "scripts" / "run_live.py"
    argv = [str(script), "--source", args.source, "--config", args.config]
    if args.no_voice:
        argv.append("--no-voice")
    if args.headless:
        argv.append("--headless")
    sys.argv = argv
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_live  # noqa: PLC0415

    return run_live.main()


def cmd_try(args) -> int:
    """Fetch (if a URL), scan, and then coach. The demo path."""
    cfg = load_config(args.config)
    target = args.target

    if target.startswith(("http://", "https://", "www.")):
        from .ingest.download import download

        print(f"fetching {target} ...")
        path = download(
            target,
            resolve_path(cfg, "paths.raw"),
            max_height=cfg_get(cfg, "ingest.download.max_height"),
            max_duration_s=None if args.any_length
            else cfg_get(cfg, "ingest.download.max_duration_s"),
        )
        print(f"  -> {path}")
    else:
        path = Path(target)
        if not path.exists():
            print(f"{path} does not exist", file=sys.stderr)
            return 1

    scan_args = argparse.Namespace(video=str(path), config=args.config, force=False)
    cmd_scan(scan_args)

    coach_args = argparse.Namespace(
        source=str(path), config=args.config,
        no_voice=args.no_voice, headless=args.headless,
    )
    return cmd_coach(coach_args)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deadbug", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download a source video")
    p.add_argument("url")
    p.add_argument("--probe-only", action="store_true", help="metadata only, no download")
    p.add_argument("--any-length", action="store_true", help="ignore the duration cap")
    p.add_argument("--force", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("dedup", help="find clips that are the same footage")
    _add_common(p)
    p.set_defaults(func=cmd_dedup)

    p = sub.add_parser("clips-template", help="write the data/clips.csv skeleton")
    p.add_argument("--force", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_clips_template)

    p = sub.add_parser("scan", help="does this video contain any exercise?")
    p.add_argument("video")
    p.add_argument("--force", action="store_true", help="re-extract instead of using the cache")
    _add_common(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("extract", help="pose + mask for every clip in the manifest")
    p.add_argument("--force", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("build", help="build the reps table")
    p.add_argument("--force", action="store_true", help="re-extract every clip")
    _add_common(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("qc", help="write reports/qc.csv and qc.html")
    _add_common(p)
    p.set_defaults(func=cmd_qc)

    p = sub.add_parser("band", help="fit the LOSO normative band")
    _add_common(p)
    p.set_defaults(func=cmd_band)

    p = sub.add_parser("coach", help="run the live app")
    p.add_argument("--source", default="0", help="camera index or video path")
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--headless", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_coach)

    p = sub.add_parser("try", help="fetch (if a URL), scan, then coach")
    p.add_argument("target", help="a YouTube URL or a local video path")
    p.add_argument("--any-length", action="store_true")
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--headless", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_try)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

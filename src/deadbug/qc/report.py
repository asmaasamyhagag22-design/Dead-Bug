"""Per-clip quality control: which clips may be used, and why.

Two dispositions, and the difference between them is not severity but
recoverability:

``flag``
    The measurement is usable but degraded. Reported, kept, counted. A low
    detection rate or a wobbling camera makes the numbers noisier; it does not
    make them wrong.

``reject``
    A precondition of the measurement is false, so the numbers are not noisy --
    they are meaningless. Two cases only: the floor estimate failed, which makes
    every lumbar reading arbitrary, and too few reps, which makes a per-clip
    statistic an anecdote.

The thresholds all live in ``configs/base.yaml`` under ``qc:``. None of them are
tuned against an outcome; they are stated in advance, which is what lets a
rejected clip be reported as a rejected clip rather than quietly dropped.

The output is deliberately two files. ``reports/qc.csv`` is the machine-readable
record that :mod:`deadbug.dataset.build` and the splits can read; ``reports/qc.html``
is what a human actually looks at before deciding a clip is worth re-filming.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

#: Sentinel distinguishing "the key was never supplied" from "the value is
#: None". Both are failures, but they are different failures: a missing key
#: means the gate never ran, and a None means the clip had nothing to measure.
_ABSENT = object()

#: Checks that downgrade a clip to ``flag``. Every name here must be a key that
#: :func:`deadbug.dataset.build.build_clip` actually puts in its diagnostics --
#: a gate wired to a key nobody produces reports ``:missing`` on every clip and
#: silently stops being a gate. :func:`assert_gates_wired` is the guard.
FLAG_CHECKS = ("detection_rate", "mean_visibility_core", "torso_len_cv", "alternation_intact")

#: Checks that reject it outright.
REJECT_CHECKS = ("floor_inlier_ratio", "rep_count")


def assert_gates_wired(measures: dict) -> list[str]:
    """Return the flag gates that this diagnostics dict cannot feed.

    Called by :func:`run` so a mis-wired gate fails loudly once, rather than
    quietly emitting ``<name>:missing`` on every clip forever. The floor and
    rep-count keys are checked separately because they are nested / derived.
    """
    return [name for name in FLAG_CHECKS if name not in measures]


@dataclass
class ClipQC:
    """One clip's QC verdict, with every measured value kept alongside it."""

    clip_id: str
    status: str = "ok"
    measures: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    rejects: list[str] = field(default_factory=list)

    #: Columns owned by the verdict. A diagnostics key of the same name is
    #: prefixed rather than allowed to overwrite it -- silently replacing
    #: ``status`` with something a measurement happened to be called would put a
    #: wrong verdict in qc.csv with nothing to show for it.
    RESERVED = ("clip_id", "status", "flags", "rejects")

    def as_row(self) -> dict[str, Any]:
        row = {"clip_id": self.clip_id, "status": self.status,
               "flags": ";".join(self.flags), "rejects": ";".join(self.rejects)}
        for key, value in self.measures.items():
            row[f"measure_{key}" if key in self.RESERVED else key] = _round(value)
        return row


def evaluate_clip(measures: dict[str, Any], cfg: dict, clip_id: str = "") -> ClipQC:
    """Apply the configured gates to one clip's measurements.

    ``measures`` is the diagnostics dict from
    :func:`deadbug.dataset.build.build_clip`, optionally extended. A missing
    measurement is **not** a pass: the check is recorded as ``<name>:missing``
    and flagged, because a gate that silently skips is a gate that is not there.
    """
    from ..config import cfg_get

    qc = ClipQC(clip_id=clip_id or str(measures.get("clip_id", "")), measures=dict(measures))

    _gate_min(qc, "detection_rate", cfg_get(cfg, "qc.min_detection_rate"), "flag")
    _gate_min(qc, "mean_visibility_core", cfg_get(cfg, "qc.min_mean_visibility_core"), "flag")
    _gate_max(qc, "torso_len_cv", cfg_get(cfg, "qc.max_torso_len_cv"), "flag")

    if cfg_get(cfg, "qc.require_alternation"):
        # build.py sets this to None when a clip produced no reps at all --
        # check_alternation returns intact vacuously for 0 or 1 rep, so an
        # empty clip must not be recorded as having passed the gate.
        intact = measures.get("alternation_intact", _ABSENT)
        if intact is _ABSENT:
            qc.flags.append("alternation:missing")
        elif intact is None:
            qc.flags.append("alternation:no_reps")
        elif not intact:
            qc.flags.append("alternation:broken")

    floor = measures.get("floor") or {}
    ratio = floor.get("inlier_ratio", measures.get("floor_inlier_ratio"))
    minimum = cfg_get(cfg, "qc.min_floor_inlier_ratio")
    if ratio is None:
        qc.rejects.append("floor_inlier_ratio:missing")
    elif not np.isfinite(ratio) or ratio < minimum:
        qc.rejects.append(f"floor_inlier_ratio={_round(ratio)}<{minimum}")

    n_reps = measures.get("n_reps")
    min_reps = cfg_get(cfg, "qc.min_rep_count")
    if n_reps is None:
        qc.rejects.append("rep_count:missing")
    elif n_reps < min_reps:
        qc.rejects.append(f"rep_count={n_reps}<{min_reps}")

    qc.status = "reject" if qc.rejects else ("flag" if qc.flags else "ok")
    return qc


def _gate_min(qc: ClipQC, key: str, minimum: float, disposition: str) -> None:
    value = qc.measures.get(key)
    target = qc.flags if disposition == "flag" else qc.rejects
    if value is None:
        target.append(f"{key}:missing")
    elif not np.isfinite(value) or value < minimum:
        target.append(f"{key}={_round(value)}<{minimum}")


def _gate_max(qc: ClipQC, key: str, maximum: float, disposition: str) -> None:
    value = qc.measures.get(key)
    target = qc.flags if disposition == "flag" else qc.rejects
    if value is None:
        target.append(f"{key}:missing")
    elif not np.isfinite(value) or value > maximum:
        target.append(f"{key}={_round(value)}>{maximum}")


def core_visibility(kpts_raw: np.ndarray) -> float:
    """Mean visibility of shoulders and hips, from the raw mp33 channel 3.

    Computed on **raw** keypoints: normalization does not touch the visibility
    channel, but reading it off a normalized array invites someone to pass a
    COCO-17 projection, which has dropped it.
    """
    from ..pose import skeleton as sk

    arr = np.asarray(kpts_raw, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 4:
        return float("nan")
    values = arr[:, list(sk.CORE_JOINTS_MP33), 3]
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def evaluate_all(diagnostics: Iterable[dict], cfg: dict) -> list[ClipQC]:
    return [evaluate_clip(d, cfg, clip_id=str(d.get("clip_id", ""))) for d in diagnostics]


def summarise(results: list[ClipQC]) -> dict:
    counts = {"ok": 0, "flag": 0, "reject": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {
        "n_clips": len(results),
        "counts": counts,
        "usable": [r.clip_id for r in results if r.status != "reject"],
        "rejected": {r.clip_id: r.rejects for r in results if r.status == "reject"},
    }


def write_csv(results: list[ClipQC], path: str | Path) -> Path:
    import csv

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.as_row() for r in results]
    # Union of keys, stable: the fixed columns first, then whatever the
    # diagnostics happened to carry, so a new measurement shows up without a
    # code change here.
    fixed = ["clip_id", "status", "flags", "rejects"]
    extra = sorted({k for row in rows for k in row} - set(fixed))
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fixed + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out


_STATUS_COLOUR = {"ok": "#1a7f37", "flag": "#9a6700", "reject": "#cf222e"}


def write_html(results: list[ClipQC], path: str | Path, title: str = "Dead Bug QC") -> Path:
    """A single self-contained page. No assets, no CDN -- it has to open from a USB stick."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = summarise(results)

    rows = []
    for r in sorted(results, key=lambda x: (x.status != "reject", x.clip_id)):
        colour = _STATUS_COLOUR.get(r.status, "#57606a")
        notes = "<br>".join(html.escape(n) for n in (r.rejects + r.flags)) or "&ndash;"
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(r.clip_id)}</code></td>"
            f'<td style="color:{colour};font-weight:600">{r.status}</td>'
            f"<td>{_cell(r.measures.get('detection_rate'))}</td>"
            f"<td>{_cell((r.measures.get('floor') or {}).get('inlier_ratio'))}</td>"
            f"<td>{_cell(r.measures.get('view_score'))}</td>"
            f"<td>{_cell(r.measures.get('torso_len_cv'))}</td>"
            f"<td>{r.measures.get('n_reps', '&ndash;')}</td>"
            f"<td>{notes}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 68rem; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border-bottom: 1px solid #d0d7de; padding: .4rem .6rem; text-align: left;
          vertical-align: top; }}
 th {{ background: #f6f8fa; }}
 code {{ font-size: .9em; }}
 .sum {{ background: #f6f8fa; padding: .8rem 1rem; border-radius: 6px; }}
</style>
<h1>{html.escape(title)}</h1>
<p class="sum">{summary['n_clips']} clips &mdash;
 <b>{summary['counts'].get('ok', 0)}</b> ok,
 <b>{summary['counts'].get('flag', 0)}</b> flagged,
 <b>{summary['counts'].get('reject', 0)}</b> rejected.</p>
<p><small>A <b>flag</b> means the measurement is degraded but usable. A
<b>reject</b> means a precondition failed &mdash; the floor could not be
estimated, or the clip has too few reps &mdash; so the numbers would not be
noisy, they would be meaningless.</small></p>
<table>
<tr><th>clip</th><th>status</th><th>detection</th><th>floor inliers</th>
    <th>view score</th><th>torso CV</th><th>reps</th><th>notes</th></tr>
{chr(10).join(rows)}
</table>
"""
    out.write_text(document, encoding="utf-8")
    return out


def run(diagnostics: Iterable[dict], cfg: dict) -> dict:
    """Evaluate, write both reports, and return the summary."""
    from ..config import cfg_get, REPO_ROOT

    diagnostics = list(diagnostics)
    results = evaluate_all(diagnostics, cfg)
    csv_path = REPO_ROOT / cfg_get(cfg, "qc.csv_out")
    html_path = REPO_ROOT / cfg_get(cfg, "qc.html_out")
    write_csv(results, csv_path)
    write_html(results, html_path)

    summary = summarise(results)
    summary["csv"] = str(csv_path)
    summary["html"] = str(html_path)

    # A gate nobody can feed is not a strict gate, it is an absent one.
    unwired = sorted({g for d in diagnostics for g in assert_gates_wired(d)})
    if unwired:
        summary["unwired_gates"] = unwired
    return summary


def _round(value: Any, places: int = 4) -> Any:
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), places)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return value


def _cell(value: Any) -> str:
    rounded = _round(value)
    return "&ndash;" if rounded is None else html.escape(str(rounded))

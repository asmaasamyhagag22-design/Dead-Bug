"""The two tables this project stores, and the rules they must satisfy.

Everything downstream -- the LOSO splits, the normative band, the QC report --
reads one of two tables:

``data/clips.csv``
    One row per **video file**. Written by hand, because the two fields that
    matter most cannot be derived from the pixels: ``person_id`` (who is in the
    clip) and ``condition`` (what they were trying to do). See
    :func:`validate_clips`.

reps table (``data/processed/reps.parquet``)
    One row per **single-side extension**, built by :mod:`deadbug.dataset.build`.
    Every row carries its ``clip_id`` and therefore its ``person_id``, which is
    what makes a leakage assertion possible at all.

Two invariants are enforced here rather than left to a convention:

**Same dedup group => same person.** :mod:`deadbug.ingest.dedup` already found
that ``clip.mp4`` and ``videoplayback (4).mp4`` are byte-identical and that the
three ``Video Project 2*`` exports share 44-82% of their frames. If those were
given different ``person_id`` values, LOSO would put the same person on both
sides of a split and every number after that would be optimistic. The check is
cheap and the failure is invisible, which is exactly the combination that
justifies enforcing it in code.

**The label may not come from the signal.** ``condition`` must have
``label_source="intent"`` -- recorded from what the subject was asked to do
before filming. Deriving it from ``lumbar_gap`` and then feeding ``lumbar_gap``
to a classifier measures nothing but the threshold that produced the labels.
:func:`validate_clips` rejects any other source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------

#: What the subject was asked to do. ``correct`` is the only condition the
#: normative band is built from; the other three are the deliberate faults from
#: the filming protocol (HANDOVER section 8).
CONDITIONS: tuple[str, ...] = ("correct", "arched", "fast", "rotated")

#: Camera obliqueness bucket, from :func:`deadbug.geometry.normalize.classify_view`.
#: Only ``side`` may build the band; ``oblique45`` is test-only.
VIEWS: tuple[str, ...] = ("side", "oblique45", "other")

#: Where ``condition`` came from. Only one value is admissible -- see the module
#: docstring. The field exists so that the constraint is recorded in the data
#: rather than only in prose.
LABEL_SOURCES: tuple[str, ...] = ("intent",)

#: QC disposition. ``reject`` rows are excluded from every table downstream;
#: ``flag`` rows are kept and reported.
QC_STATUS: tuple[str, ...] = ("ok", "flag", "reject")


class SchemaError(ValueError):
    """A table violated a rule that would silently corrupt a result."""


# --------------------------------------------------------------------------
# clips.csv
# --------------------------------------------------------------------------


@dataclass
class ClipRecord:
    """One row of ``data/clips.csv``.

    ``person_id`` and ``condition`` are the human-supplied fields. Everything
    else is either bookkeeping or copied from the dedup / triage reports so the
    manifest is readable on its own.
    """

    clip_id: str
    file: str
    person_id: str
    condition: str
    view: str
    label_source: str = "intent"
    dedup_group: str = ""
    session: str = ""
    camera_angle_deg: float = float("nan")
    detection_rate: float = float("nan")
    view_score: float = float("nan")
    torso_len_cv: float = float("nan")
    qc_status: str = "ok"
    notes: str = ""

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ClipRecord":
        known = {f.name for f in fields(cls)}
        unknown = set(row) - known
        if unknown:
            raise SchemaError(f"unknown column(s) in clips.csv: {sorted(unknown)}")
        missing = {"clip_id", "file", "person_id", "condition", "view"} - set(row)
        if missing:
            raise SchemaError(f"clips.csv row missing required field(s): {sorted(missing)}")
        coerced = dict(row)
        for name in ("camera_angle_deg", "detection_rate", "view_score", "torso_len_cv"):
            if name in coerced:
                coerced[name] = _as_float(coerced[name])
        return cls(**coerced)

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def validate_clips(records: Iterable[ClipRecord | dict]) -> list[ClipRecord]:
    """Check a clip manifest, raising :class:`SchemaError` on the first problem.

    Returns the validated records so the call can be used inline. The ordering
    that matters is that the leakage check runs last, on a table already known
    to be well-formed.
    """
    clips = [r if isinstance(r, ClipRecord) else ClipRecord.from_row(r) for r in records]
    if not clips:
        raise SchemaError("clip manifest is empty")

    seen: set[str] = set()
    for clip in clips:
        if not clip.clip_id:
            raise SchemaError(f"empty clip_id for file {clip.file!r}")
        if clip.clip_id in seen:
            raise SchemaError(f"duplicate clip_id: {clip.clip_id!r}")
        seen.add(clip.clip_id)

        if not clip.person_id:
            raise SchemaError(
                f"{clip.clip_id}: person_id is blank. It cannot be inferred -- "
                "assign it by hand from reports/dedup.csv and the preview videos."
            )
        _require(clip.condition, CONDITIONS, "condition", clip.clip_id)
        _require(clip.view, VIEWS, "view", clip.clip_id)
        _require(clip.qc_status, QC_STATUS, "qc_status", clip.clip_id)

        if clip.label_source not in LABEL_SOURCES:
            raise SchemaError(
                f"{clip.clip_id}: label_source={clip.label_source!r}. Only 'intent' is "
                "admissible -- a condition derived from the signal makes any model "
                "trained on that signal circular."
            )

    _check_dedup_consistency(clips)
    return clips


def _check_dedup_consistency(clips: list[ClipRecord]) -> None:
    """Same dedup group => same person. See the module docstring."""
    by_group: dict[str, set[str]] = {}
    for clip in clips:
        if clip.dedup_group:
            by_group.setdefault(clip.dedup_group, set()).add(clip.person_id)
    for group, people in sorted(by_group.items()):
        if len(people) > 1:
            raise SchemaError(
                f"dedup group {group} spans person_id {sorted(people)}. The dedup "
                "report says these clips are the same footage, so treating them as "
                "different subjects would leak one person across a LOSO split."
            )


def read_clips(path: str | Path) -> list[ClipRecord]:
    """Load and validate ``data/clips.csv``.

    Uses :mod:`csv` rather than pandas so that this module stays importable in
    the Track A environment, which pins a different pandas.
    """
    import csv

    p = Path(path)
    if not p.exists():
        raise SchemaError(
            f"{p} does not exist. It is written by hand from reports/dedup.csv -- "
            "run `deadbug clips-template` to generate a skeleton."
        )
    with open(p, "r", encoding="utf-8", newline="") as fh:
        rows = [
            {k: v for k, v in row.items() if k and v not in (None, "")}
            for row in csv.DictReader(fh)
        ]
    if not rows:
        raise SchemaError(f"{p} has a header but no rows")
    return validate_clips(rows)


def write_clips(records: Iterable[ClipRecord], path: str | Path) -> Path:
    """Write a clip manifest with a stable column order."""
    import csv

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = ClipRecord.field_names()
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())
    return out


# --------------------------------------------------------------------------
# reps table
# --------------------------------------------------------------------------


@dataclass
class RepRecord:
    """One single-side extension, with its clip provenance attached.

    The provenance fields are duplicated onto every rep on purpose. A reps table
    that has to be joined back to ``clips.csv`` before it can be split is a table
    that will eventually be split without the join.
    """

    rep_id: str
    clip_id: str
    person_id: str
    condition: str
    view: str
    side: str
    rep_index: int
    start_s: float
    end_s: float
    peak_s: float
    duration_s: float
    t_extend_s: float
    t_return_s: float
    dwell_s: float
    overlap_s: float
    excursion_peak: float
    lumbar_gap_mean: float = float("nan")
    lumbar_gap_peak: float = float("nan")
    lumbar_gap_at_excursion_peak: float = float("nan")
    rib_gap_peak: float = float("nan")
    rot_dev_peak: float = float("nan")
    sparc_wrist: float = float("nan")
    sparc_ankle: float = float("nan")
    flags: str = ""

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


#: Columns a model may read.
FEATURE_COLUMNS: tuple[str, ...] = (
    "duration_s", "t_extend_s", "t_return_s", "dwell_s", "overlap_s",
    "excursion_peak", "lumbar_gap_mean", "lumbar_gap_peak",
    "lumbar_gap_at_excursion_peak", "rib_gap_peak", "rot_dev_peak",
    "sparc_wrist", "sparc_ankle",
)

#: Columns that identify a row or its subject. Never features.
ID_COLUMNS: tuple[str, ...] = (
    "rep_id", "clip_id", "person_id", "condition", "view", "side", "rep_index",
)


def assert_not_circular(feature_columns: Iterable[str], label_column: str) -> None:
    """Refuse a setup where the label was computed from one of the features.

    The specific trap this exists for: labelling a rep ``arched`` by thresholding
    ``lumbar_gap`` and then handing ``lumbar_gap_peak`` to a classifier. The
    resulting accuracy measures the threshold, not the exercise.
    """
    if label_column in set(feature_columns):
        raise SchemaError(
            f"label column {label_column!r} is also a feature -- circular by construction"
        )
    if label_column.startswith("lumbar_gap"):
        raise SchemaError(
            f"label column {label_column!r} is derived from the lumbar signal. "
            "Labels must come from filming intent (label_source='intent')."
        )


def validate_reps(
    records: Iterable[RepRecord | dict], clips: Iterable[ClipRecord]
) -> list[RepRecord]:
    """Check that every rep points at a clip in the manifest, consistently."""
    by_id = {c.clip_id: c for c in clips}
    reps = [r if isinstance(r, RepRecord) else RepRecord(**r) for r in records]

    seen: set[str] = set()
    for rep in reps:
        if rep.rep_id in seen:
            raise SchemaError(f"duplicate rep_id: {rep.rep_id!r}")
        seen.add(rep.rep_id)

        clip = by_id.get(rep.clip_id)
        if clip is None:
            raise SchemaError(f"{rep.rep_id}: clip_id {rep.clip_id!r} is not in the manifest")
        for attr in ("person_id", "condition", "view"):
            if getattr(rep, attr) != getattr(clip, attr):
                raise SchemaError(
                    f"{rep.rep_id}: {attr}={getattr(rep, attr)!r} disagrees with "
                    f"clip {rep.clip_id} ({getattr(clip, attr)!r})"
                )
        if rep.side not in ("R", "L"):
            raise SchemaError(f"{rep.rep_id}: side must be 'R' or 'L', got {rep.side!r}")
        if not (rep.end_s > rep.start_s):
            raise SchemaError(f"{rep.rep_id}: end_s must exceed start_s")
    return reps


def _require(value: str, allowed: tuple[str, ...], name: str, where: str) -> None:
    if value not in allowed:
        raise SchemaError(f"{where}: {name}={value!r} not in {list(allowed)}")


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

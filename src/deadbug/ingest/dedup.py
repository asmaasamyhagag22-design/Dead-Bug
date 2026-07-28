"""Detect duplicate and nested-edit clips before they corrupt the splits.

Subject identity is the one thing that must never leak across a
leave-one-subject-out split, and the same person can reach the dataset more
than once: an exact re-download, a re-encode, a screen recording of a YouTube
playback, or an editor export that contains another clip as a sub-range. Two
copies of one person counted as two subjects would inflate every number
silently, and no test downstream would catch it.

Three tiers, cheapest first:

1. **md5** -- byte-identical files. Free, and it already found one pair here.
2. **dHash** on frames sampled once per *second* -- survives re-encoding, mild
   rescaling and letterboxing.
3. **subset search** -- slides one clip's fingerprint along another's to find a
   clip contained inside a longer edit.

The output is a suggestion, not a verdict. A human assigns ``person_id`` in
``data/clips.csv`` after looking at the preview videos; automating identity is
how a wrong merge gets in without anyone noticing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

DHASH_GRID = (9, 8)      # 9x8 grayscale -> 64-bit hash
SAMPLE_EVERY_S = 1.0
HAMMING_THRESHOLD = 8

#: Above this, merge automatically -- the clips are the same footage.
MATCH_FRAC_THRESHOLD = 0.85

#: Between REVIEW and MATCH, flag for a human instead of guessing. Measured on
#: this dataset the gap is unambiguous: unrelated clips score exactly 0.00,
#: while the three ``Video Project 2*`` exports score 0.24-0.82 against each
#: other and 0.00 against everything else. They are one session in three edits.
#: Auto-merging at 0.24 would be reckless on a different dataset; ignoring it
#: would silently count one person as three subjects. So: flag it, and let the
#: human who assigns person_id decide.
REVIEW_FRAC_THRESHOLD = 0.15


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """md5 of the file bytes -- catches exact duplicates only."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _dhash(frame_bgr: np.ndarray, grid: tuple[int, int] = DHASH_GRID) -> np.uint64:
    """Difference hash: is each pixel brighter than the one to its right?"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, grid, interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, :-1] > small[:, 1:]).flatten()
    value = np.uint64(0)
    for bit in bits:
        value = np.uint64(value << np.uint64(1)) | np.uint64(bool(bit))
    return value


def video_fingerprint(
    path: str | Path, sample_every_s: float = SAMPLE_EVERY_S
) -> tuple[np.ndarray, np.ndarray]:
    """``(hashes, timestamps_s)`` sampling on the **time** axis.

    Time, not frame index -- these clips run at 23.98, 25, 29.97 and 30 fps, so
    a fixed frame stride would sample different moments from two copies of the
    same footage and the comparison would fail on re-encodes.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(sample_every_s * fps)))
        hashes, stamps = [], []
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                hashes.append(_dhash(frame))
                stamps.append(index / fps)
            index += 1
    finally:
        cap.release()
    return np.array(hashes, dtype=np.uint64), np.array(stamps, dtype=np.float64)


def hamming(a: np.uint64, b: np.uint64) -> int:
    return int(bin(int(a) ^ int(b)).count("1"))


def _match_counts(fp_a: np.ndarray, fp_b: np.ndarray, threshold: int) -> np.ndarray:
    """For each hash in A, whether any hash in B is within ``threshold`` bits."""
    return np.array(
        [any(hamming(x, y) <= threshold for y in fp_b) for x in fp_a], dtype=bool
    )


def pairwise_similarity(
    fp_a: np.ndarray, fp_b: np.ndarray, threshold: int = HAMMING_THRESHOLD
) -> float:
    """Fraction of A's sampled frames that appear somewhere in B."""
    if fp_a.size == 0 or fp_b.size == 0:
        return 0.0
    return float(_match_counts(fp_a, fp_b, threshold).mean())


def find_subset(
    fp_a: np.ndarray, ts_a: np.ndarray, fp_b: np.ndarray, ts_b: np.ndarray,
    threshold: int = HAMMING_THRESHOLD, min_matched_s: float = 5.0,
) -> tuple[float, float] | None:
    """Locate A inside B, returning ``(start_s, end_s)`` in B's timeline.

    This is what catches the nested-edit case: three exports of one recording,
    where the shortest is a sub-range of the longest.
    """
    if fp_a.size == 0 or fp_b.size == 0:
        return None
    matched_in_b = [
        int(np.argmin([hamming(x, y) for y in fp_b]))
        if min(hamming(x, y) for y in fp_b) <= threshold else -1
        for x in fp_a
    ]
    hits = [i for i in matched_in_b if i >= 0]
    if len(hits) < 2:
        return None
    span = ts_b[max(hits)] - ts_b[min(hits)]
    if span < min_matched_s or len(hits) / fp_a.size < 0.5:
        return None
    return float(ts_b[min(hits)]), float(ts_b[max(hits)])


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_dedup_groups(
    paths: list[str | Path],
    sample_every_s: float = SAMPLE_EVERY_S,
    threshold: int = HAMMING_THRESHOLD,
    match_frac: float = MATCH_FRAC_THRESHOLD,
) -> dict:
    """Group clips that appear to show the same recording.

    Returns ``{"groups": {path: "G01"}, "evidence": [...]}`` where each evidence
    row records *why* two clips were merged, so the decision is auditable rather
    than a black box.
    """
    paths = [Path(p) for p in paths]
    hashes = {p: file_hash(p) for p in paths}
    fingerprints = {}
    for p in paths:
        try:
            fingerprints[p] = video_fingerprint(p, sample_every_s)
        except OSError:
            fingerprints[p] = (np.array([], dtype=np.uint64), np.array([]))

    uf = _UnionFind(paths)
    evidence = []

    for i, a in enumerate(paths):
        for b in paths[i + 1 :]:
            if hashes[a] == hashes[b]:
                uf.union(a, b)
                evidence.append({"a": a.name, "b": b.name, "merged": True,
                                 "reason": "md5 identical",
                                 "detail": hashes[a][:12], "similarity": 1.0})
                continue

            fa, ta = fingerprints[a]
            fb, tb = fingerprints[b]
            sim_ab = pairwise_similarity(fa, fb, threshold)
            sim_ba = pairwise_similarity(fb, fa, threshold)
            best = max(sim_ab, sim_ba)
            if best < REVIEW_FRAC_THRESHOLD:
                continue

            # The shorter clip is the one that could be contained in the longer.
            short, long_ = (a, b) if fa.size <= fb.size else (b, a)
            window = find_subset(*fingerprints[short], *fingerprints[long_], threshold)
            detail = (
                f"{short.name} spans {window[0]:.1f}-{window[1]:.1f}s of {long_.name}"
                if window else "shared footage"
            )
            if best >= match_frac:
                uf.union(a, b)
                evidence.append({"a": a.name, "b": b.name, "merged": True,
                                 "reason": "perceptual match", "detail": detail,
                                 "similarity": round(best, 3)})
            else:
                evidence.append({"a": a.name, "b": b.name, "merged": False,
                                 "reason": "REVIEW -- likely same session",
                                 "detail": detail, "similarity": round(best, 3)})

    roots = {}
    groups = {}
    for p in paths:
        root = uf.find(p)
        if root not in roots:
            roots[root] = f"G{len(roots) + 1:02d}"
        groups[p] = roots[root]
    return {"groups": groups, "evidence": evidence, "hashes": hashes}


def write_report(result: dict, out_csv: str | Path) -> Path:
    """Write the audit trail from raw file to suggested group.

    Every merge and every flagged pair is listed with its evidence, so the
    ``person_id`` assignment in ``clips.csv`` can be traced back to a reason.
    """
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    by_group: dict[str, list[Path]] = {}
    for path, group in result["groups"].items():
        by_group.setdefault(group, []).append(path)

    notes: dict[str, list[str]] = {}
    for row in result["evidence"]:
        tag = "MERGED" if row.get("merged") else "REVIEW"
        text = f"{tag} with {row['a']}: {row['reason']} (sim={row['similarity']})"
        notes.setdefault(row["b"], []).append(text)
        notes.setdefault(row["a"], []).append(
            f"{tag} with {row['b']}: {row['reason']} (sim={row['similarity']})"
        )

    lines = ["file,dedup_group,md5,action,note"]
    for group in sorted(by_group):
        for path in sorted(by_group[group]):
            note = "; ".join(notes.get(path.name, []))
            action = (
                "duplicate" if len(by_group[group]) > 1
                else ("review" if "REVIEW" in note else "unique")
            )
            lines.append(
                f'"{path.name}",{group},{result["hashes"][path][:12]},{action},"{note}"'
            )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def similarity_matrix(
    paths: list[str | Path], sample_every_s: float = SAMPLE_EVERY_S,
    threshold: int = HAMMING_THRESHOLD,
) -> tuple[np.ndarray, list[str]]:
    """Full pairwise similarity, for the human who assigns ``person_id``.

    Reading the matrix is more informative than reading a threshold's verdict:
    on this dataset unrelated clips score exactly 0.00, which makes any non-zero
    entry worth a look.
    """
    paths = [Path(p) for p in paths]
    fingerprints = {p: video_fingerprint(p, sample_every_s) for p in paths}
    n = len(paths)
    matrix = np.eye(n)
    for i, a in enumerate(paths):
        for j, b in enumerate(paths):
            if i != j:
                matrix[i, j] = pairwise_similarity(
                    fingerprints[a][0], fingerprints[b][0], threshold
                )
    return matrix, [p.name for p in paths]

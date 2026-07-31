# Datasheet — Dead Bug clips

Follows the structure of *Datasheets for Datasets* (Gebru et al.). Two distinct
collections are used and they are kept strictly apart:

- **Track A** — Rehab-Pile, a published benchmark. Not collected here.
- **Track B** — the Dead Bug clips. The original contribution, and the subject of
  most of this document.

---

## Motivation

**Why was this collected?** To measure whether a silhouette-derived lumbar signal
can distinguish a correct Dead Bug from a faulted one, and to drive a live
coaching application.

**Who collected it?** The project author, as coursework. No funding, no
institutional sponsor.

**Was there an ethics review?** No. No patient population, no clinical setting,
no third-party recruitment. The recorded footage is of the author.

---

## Composition — Track B

### What is in it

Ten source video files in `data/clips/`, ~58 MB total, in three groups:

| origin | files | what they are |
|---|---|---|
| Public instructional video | `videoplayback (1)`, `(3)`, `(4)`, `(6)` | Dead Bug demonstrations posted publicly |
| Screen recordings | `Recording 2026-07-25 *` | captures of playback |
| Author's edits | `Video Project 2`, `2 (2)`, `2mk` | three exports of one session |
| Derived | `clip.mp4` | **byte-identical to `videoplayback (4)`** |

### Duplication, and why it is dangerous

Subject identity is the one thing that must never cross a leave-one-subject-out
split. The same person reaches the dataset more than once here, through exact
re-download, re-encoding, screen recording, and editor exports containing each
other as sub-ranges. Two copies of one person counted as two subjects would
inflate every number silently.

`ingest/dedup.py` runs three tiers — md5, then dHash sampled once per *second*
(time, not frame index: these clips run at 23.98 / 25 / 29.97 / 30 fps), then a
subset search that slides one fingerprint along another. Measured result:

| pair | similarity | disposition |
|---|---|---|
| `clip.mp4` ↔ `videoplayback (4).mp4` | 1.00, md5 identical | **merged automatically** |
| `Video Project 2 (2)` ↔ `2mk` | 0.82 | flagged for review |
| `Video Project 2` ↔ `2mk` | 0.82 | flagged for review |
| `Video Project 2` ↔ `2 (2)` | 0.44 | flagged for review |
| everything else | **0.00** | unrelated |

The gap is unambiguous on this dataset: unrelated clips score *exactly* zero.
Auto-merging at 0.44 would be reckless on a different dataset; ignoring it would
count one session as three subjects. So the tool **flags and does not decide** —
`person_id` is assigned by a human in `data/clips.csv`, and
`dataset/schema.py` refuses any manifest where one dedup group spans two people.

The audit trail is `reports/dedup.csv`: every merge and every flag with its
evidence.

### Labels

**Every Dead Bug clip currently held is labelled `correct`.** There is no faulty
footage. This is the project's central data limitation and the reason for the
two-track design — see [LIMITATIONS.md](LIMITATIONS.md) §1.

`condition` is required to have `label_source = "intent"`: it records what the
subject was *asked* to do before filming. A label derived from `lumbar_gap` and
then used to train on `lumbar_gap` measures the threshold that made the labels
and nothing else. The schema rejects any other source.

### Per-clip measurements

`reports/triage.csv` carries detection rate, median `view_score` (pixel space),
torso-length CV, core-joint visibility, and the exercise fraction, for every
clip. Highlights that shaped the project:

| clip | detection | view score | torso CV | verdict |
|---|---|---|---|---|
| `videoplayback (3)` | 92.7% | 0.119 | 0.020 | clean side view — and **0 reps in 98 s** |
| `videoplayback (6)` | 90.1% | 0.112 | 0.495 | camera zooms |
| `videoplayback (1)` | 97.8% | 0.123 | 0.274 | oblique, subject moves |
| `Video Project 2mk` | 17.3% | 0.103 | 0.020 | **detection collapses** |
| `clip` ≡ `videoplayback (4)` | 100% | 0.154 | 0.029 | duplicate |

That `videoplayback (3)` is simultaneously the cleanest camera work and
completely useless is the most informative row in the table: **instructional
videos are a source of coaching, not a source of reps.**

### What is *not* in it

- No demographic attributes. Age, sex, body composition and clothing are neither
  recorded nor inferable, so no claim can be made about how the signal behaves
  across body types — and a silhouette-based signal certainly does vary with
  clothing.
- No clinical or injury information.
- No faulted performances (§Labels).
- No footage at known camera angles, which is why `view_score` thresholds remain
  uncalibrated.

---

## Collection process — Track B

**How was it acquired?** Public instructional videos were downloaded;
`ingest/download.py` fetches a *progressive* mp4 so that ffmpeg is never invoked.
The author's clips were recorded on a phone.

**Was anything re-encoded?** No, deliberately. Sources arrive at 23.98 / 25 /
29.97 / 30 fps; rather than force a constant rate through ffmpeg, the true fps is
carried and the **signals** are resampled after extraction. Sub-clipping is a
seek inside `VideoSource`, not a new file.

**Was anything modified?** Frame dimensions are snapped to a multiple of 16
before inference. This is not cosmetic: MediaPipe 0.10.35 hard-aborts the
process — native crash, no traceback — when reading a segmentation mask from a
frame whose row stride is not 4-byte aligned (width 1006 → 4024 bytes/row).

**Sampling strategy?** Convenience sampling. There was no protocol, which is
itself a limitation.

---

## Uses

**Used for so far:** establishing instrument validity from correct-only footage;
rep-segmentation cross-validation; the live coaching demo.

**Should not be used for:** training or evaluating a fault classifier. There are
no faults in it, and a classifier trained on correct-only data with signal-derived
labels is circular by construction. `schema.assert_not_circular` blocks the
specific case.

**Not suitable for** any clinical inference, any demographic generalisation, or
any exercise other than Dead Bug.

---

## Distribution and maintenance

The clips are **not** in the git repository — `.gitignore` excludes
`data/clips/`, and the author's own footage is not redistributed. Public source
videos remain under their original terms; nothing here re-licenses them.

`data/clips/` has no second copy anywhere. It is the one directory that must be
carried by hand when moving machines.

**Errata / updates.** The intended next revision is the filming session described
in [LIMITATIONS.md](LIMITATIONS.md) §1: four sessions of ten reps (arched, fast,
rotated, correct) plus three short correct clips at 0° / 30° / 45°. That
revision would, for the first time, make `condition` non-constant and make the
camera-angle thresholds measurable.

---

## Track A — Rehab-Pile

Not collected here. 39 classification datasets across seven collections (KIMORE,
KERAAL, IRDS, KINECAL, SPHERE, UCDHE, UI-PRMD), downloaded on first run into
`data/rehabpile/` (~500 MB) and used under their original terms.

Used **only** to demonstrate that the training and evaluation code is correct on
labelled data containing real errors. Test folds hold 6–14 samples, which is why
results are reported as mean rank over datasets rather than mean score — see
`reports/masar_a_results.md`. No Rehab-Pile subject appears in Track B and no
model is transferred between the two; the shared surface is
`modeling/evaluate.py` alone, kept numpy-only so both pinned environments can
import it.

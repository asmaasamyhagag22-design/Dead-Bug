# Limitations

> **This system detects deviations from a reference movement pattern. It is not a
> diagnostic or clinical tool, was not validated on a patient population, and must
> not be used to guide medical decisions.**

Ordered by how much they constrain what may be claimed, not by how easy they are
to fix.

---

## 1. Sensitivity to real errors has not been measured

Every Dead Bug clip currently in `data/clips/` is a `correct` performance. The
project therefore separates two questions that are routinely conflated:

| | question | answered? |
|---|---|---|
| **Instrument validity** | does `lumbar_gap` actually measure back height? | ✅ yes, from correct-only footage |
| **Detector evaluation** | does it catch a real arched rep? | ❌ no faulty footage exists |

Validity is established by injecting a **synthetic** arch of known size into the
silhouette (`signals.lumbar.perturb_mask`) and checking the measured response
against the closed-form prediction `n_columns / torso_len²`. That proves the
instrument responds correctly and linearly to a lift. It says nothing about
whether a human arching their back produces a lift of a detectable size.

**What may be claimed:** the signal measures what it says it measures.
**What may not:** any sensitivity, specificity, or false-negative rate.

Filming four faulted sessions (arched / fast / rotated / correct, ten reps each)
is the single blocker. Until then the phrase "detects errors" does not belong in
a results slide.

## 2. The subject pool is too small for the numbers to be stable

Track B has a handful of subjects, and the usable subset is smaller still. Of
ten source clips (measured 31 Jul 2026, `reports/triage.csv`): one pair is
byte-identical, three are exports of one session, one collapses to 18%
detection, two have a torso-length CV above 0.10 because the camera moved, and
one contains no exercise at all.

**Exactly one clip — `videoplayback (3)` — classifies as a true side view**, and
the lumbar signal is defined only for side views. A normative band needs at least
two subjects to be leave-one-subject-out at all. Today there is one usable side
view, so the band cannot be fitted, and `dataset.normative.fit_band` raises
rather than returning something that looks like an answer.

Consequences that are not negotiable:

- **No bare percentages.** Report `0/48 reps, 95% CI [0, 7.4%], LOSO on N=6`, not
  `0% false alarms`. `modeling.evaluate.wilson_interval` exists for this.
- **N appears in the sentence**, not in a footnote.
- The normative band is built **leave-one-subject-out**
  (`dataset.normative.band_loso`). A pooled band that saw the subject it is
  judging understates their deviation by construction, and at this N the effect
  is large.

## 3. Camera-angle thresholds are uncalibrated

`triage.view_score_side_max: 0.12` and `view_score_oblique_max: 0.21` come from a
geometric table in the design document (0° → 0.00, 30° → 0.15, 45° → 0.21), not
from measurement. Body proportions move this number, so a clip near the boundary
may be filed in the wrong bucket.

The fix is three short correct clips shot at a known 0° / 30° / 45°, which is
part of the same filming session as §1. Until then, treat `side` versus
`oblique45` as a provisional label.

One thing about this measurement *is* settled: it must be computed in **pixel**
space. MediaPipe returns normalized coordinates whose x and y have different
pixel scales, and a supine subject is the worst case — the torso runs along x
while the hips separate along y. Measured on one clip: 0.240 normalized against
0.131 in pixels, enough to flip the verdict.

## 4. The lumbar signal depends on a segmentation mask, and therefore on a view

The primary signal is read off the silhouette because **MediaPipe has no landmark
anywhere on the spine** — the error the exercise is about is invisible in the
skeleton. That choice buys the signal and costs three things:

- **A side view is required.** From above or head-on, the gap between the lower
  back and the floor does not project onto the silhouette at all.
- **A visible floor line is required.** `geometry.floor` estimates it without
  calibration, but a subject on a raised mat, or a cluttered lower frame, moves
  it.
- **Loose clothing inflates the gap.** The mask follows the garment, not the
  spine. Nothing in the pipeline can distinguish a lifted back from a hanging
  t-shirt.

`signals.lumbar.geometric_signal` provides a mask-free cross-check (pelvis offset
from the shoulder–knee line), and it is genuinely weaker. It exists so that a
disagreement between the two is visible, not as a replacement.

## 5. Track A does not reproduce the published Rehab-Pile results

The benchmark ran across 39 datasets and produced a defensible table:

| model | mean rank | wins | mean macro-F1 | datasets |
|---|---|---|---|---|
| MiniRocket | **1.65** | 25 | 0.684 | 39 |
| RF (summary) | 2.17 | 11 | 0.610 | 39 |
| LITEMV | 2.67 | 1 | 0.582 | **3** |
| RF (flatten) | 2.76 | 6 | 0.528 | 39 |
| majority | 3.60 | 4 | 0.379 | 39 |

Four caveats attach to it:

- **These are library defaults, not the paper's configuration**, which could not
  be recovered. Say "ran at the library's default settings", never "reproduced
  the paper".
- **LITEMV was evaluated on a sample, on cost grounds.** One dataset with 5
  classifiers took 7365 s (~2 h) and then died on memory; MiniRocket takes
  seconds and beats it. Any LITEMV number must be quoted with its
  `n_classifiers` and `n_epochs` (5 and 300, against the library default of
  1500 epochs).
- **LITEMV runs unweighted.** `class_weight='balanced'` is set on RandomForest
  and MiniRocket; aeon does not expose it for `LITETimeClassifier`. On the
  imbalanced sets this is part of the result, not a defect to hide.
- **Ranks, not mean scores.** Test folds hold 6–14 samples, so a single fold's
  score swings on sampling alone, and averaging raw scores would let an easy
  dataset dominate. Mean rank is the standard presentation for exactly this
  reason.

Track A is evidence that the training and evaluation code is correct. It is not
evidence about Dead Bug.

## 6. Known measurement traps, all of them live

These are properties of the pipeline as it stands, not historical bugs:

| | |
|---|---|
| **Frame width must be a multiple of 16** | MediaPipe 0.10.35 hard-aborts the process — native crash, no traceback — reading a mask when the row stride is not 4-byte aligned (width 1006 → 4024 bytes/row). `ingest.video_source` snaps dimensions before inference. Any new capture path must do the same. |
| **One detector per clip** | `RunningMode.VIDEO` carries tracking state and requires increasing timestamps. Reusing a detector across clips crashes natively. |
| **`floor_inlier_ratio` is computed on the lowest slice only** | Over all columns it measures "how much of the body is on the floor", which is *anti*-correlated with correct form — the limbs are deliberately in the air. The 0.80 gate would then reject exactly the clips it exists to keep. |
| **Zero-phase filtering is offline-only** | `zero_phase=True` looks ahead. It is correct for rep-boundary timing and impossible live; the live counter is a strictly causal state machine, and the two therefore do not always agree to the frame. |

## 7. The live app judges against a per-session baseline, not a population

A session calibrates on the user's own first three reps, then judges everything
after against that. This is deliberate — there is no universal correct range of
motion, so an absolute threshold on distance-to-floor would be wrong for
everyone — but it has a real failure mode: **if the first three reps are bad, the
baseline is bad**, and the session will happily approve the same fault for the
rest of the set. The app cannot detect this on its own.

The population-level answer is the normative band of §2, which needs the subjects
of §1.

## 8. What the system does not attempt

- No injury, pathology, or clinical assessment of any kind.
- No claim about training effect, progression, or dosage.
- No multi-person handling: `pose.num_poses: 1`, and a second person in frame is
  undefined behaviour.
- No exercise other than Dead Bug. The rep definition (contralateral
  wrist-to-ankle extension) is specific to it.
- No audio, no held-out clinical validation set, no comparison against a
  motion-capture ground truth.

## 9. On instructional footage the segmenter's verdict is not stable

⚠️ **An earlier claim in HANDOVER — that `videoplayback (3).mp4` contains zero
reps across 98 seconds — does not reproduce with the code as it stands.** Re-run
on 31 Jul 2026, that clip yields 9 detections covering 90% of its duration.

The instability traces to one constant. `segment/activity.py` states the
physiological prior in its docstring as *"a dead bug extension takes roughly one
to four seconds"* and then sets `MAX_EXTEND_S = 6.0`, which is not that. The
count depends heavily on which of the two you believe:

| clip | `max_extend_s = 6.0` (code) | `= 4.0` (docstring) | `= 3.0` |
|---|---|---|---|
| `videoplayback (3)` | 9 reps, 90% | 3 reps, 30% | 0 reps, 0% |
| `videoplayback (1)` | 6 reps, 70% | 6 reps, 70% | 6 reps, 70% |
| `clip` ≡ `videoplayback (4)` | 4 reps, 100% | 4 reps, 100% | 0 reps, 0% |
| `Recording …171515` | 0 reps | 0 reps | 0 reps |

The two clips with real reps are stable across the range; the instructional clip
is the one that swings. And the 9 detections at 6.0 s look like what they are —
extension times of 3.6–6.0 s against a coached tempo of about 1–2 s, and an
alternation string of `LRLLRLRLR` when a Dead Bug alternates strictly. Those are
the coach drifting between demonstrations, not reps.

**What may be said:** the segmenter separates the two clips that contain real
sets from the one that does not, and its verdict on demonstration footage is
sensitive to a tempo bound the codebase has not settled. **What may not be
said:** that any clip was proven to contain zero reps, or that two independent
methods confirmed it. The live counter and the offline segmenter now share the
same 6.0 s bound, so agreement between them is not independent evidence.

This does not weaken the case for filming — it strengthens it. A tempo threshold
cannot be calibrated on footage where nobody recorded what the true count was.
Ten deliberate reps with a known answer settles it in one take.

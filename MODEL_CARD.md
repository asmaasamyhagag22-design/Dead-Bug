# Model Card — Dead Bug AQA

> **This system detects deviations from a reference movement pattern. It is not a
> diagnostic or clinical tool, was not validated on a patient population, and must
> not be used to guide medical decisions.**

## What this is

A **measurement instrument with a decision rule on top**, not a trained
classifier. That distinction is the whole design, so it is stated first.

The instrument is `lumbar_gap` — the silhouette area enclosed between the lower
back and the floor line, normalized by torso length squared. The decision rule
compares a rep's value against a **normative band**: the distribution of that
same quantity across correct reps, binned by how far the rep actually extended.

No weights are learned from Dead Bug footage. There is nothing to overfit,
because nothing was fitted. What *is* estimated from data is the band itself —
per-bin mean and standard deviation — and it is estimated leave-one-subject-out.

| | |
|---|---|
| Version | 0.1.0 |
| Status | research prototype; **detector sensitivity not yet evaluated** |
| Pose backbone | MediaPipe Pose Landmarker, `heavy`, Tasks API, `RunningMode.VIDEO` |
| Decision rule | per-excursion-bin control limit, `mean + 2σ`, LOSO-fitted |
| Learned parameters | none from Dead Bug footage |

## The signal

```
u             = unit vector hip_centre -> shoulder_centre
window        = x in [0.00, 0.35] * torso_len_px along u
gap(x)        = max(0, floor_y(x) - bottom(x))
lumbar_gap(t) = Σ gap(x) / torso_len_px²
```

**Why the silhouette and not the skeleton:** MediaPipe returns 33 landmarks and
**not one of them is on the spine**. The error a Dead Bug is about — the lower
back arching off the floor — is invisible to a keypoint model. A system built on
keypoints alone cannot see the thing the exercise is for.

**Why divide by torso_len²:** an area over a length squared is dimensionless, so
the signal does not change when the camera moves closer or the subject is taller.

**Why bin on excursion, not time:** there is no fixed correct range of motion.
Two people performing the same correct rep at different speeds occupy the same
place on an excursion axis and different places on a time axis. Binning on
excursion also closes the loophole where a rep scores well by never extending far
enough to load the back.

## Intended use

- Self-directed practice feedback for a healthy adult performing Dead Bug.
- A side-view camera, subject supine, floor visible in frame.
- Research and coursework on action quality assessment.

## Out of scope

- Any clinical, diagnostic, rehabilitative or triage purpose.
- Patient populations. No patient data was used at any point.
- Any exercise other than Dead Bug — the rep definition (contralateral
  wrist-to-opposite-ankle extension) does not transfer.
- More than one person in frame (`pose.num_poses: 1`).
- Non-side views. From head-on the gap does not project onto the silhouette.

## How a rep is judged

1. **Setup** — the subject lies still for ~3 s. The floor line is estimated from
   the lowest slice of the silhouette across those frames, then frozen.
2. **Baseline** — the first three reps establish the subject's own reference.
3. **Judgement** — every subsequent rep is scored against that baseline, or
   against the LOSO normative band when one has been fitted.

The setup hold is the one advantage the live path has over any recorded clip:
the app can ask for the calibration that a YouTube video never provides.

## Evaluation

### What has been evaluated

**Gate 0 — invariance.** Translation, scale, rotation and horizontal-flip
invariance of the normalization and skeleton layers, as unit tests with
closed-form expected values. 113 tests, all passing. If Gate 0 fails, every
number after it is meaningless, so it runs first and blocks everything.

**Instrument validity (V1, V3).** A synthetic arch of known size is injected into
the silhouette and the measured response is checked against the closed-form
prediction `n_columns / torso_len²`. Established from correct-only footage, with
no faulty reps required.

**Mirror equivalence.** The feature-space flip permutation is asserted against
the long route — mirror the raw pixels, re-normalize, re-stack. A wrong flip does
not crash; it teaches contradictory left/right labels while the loss descends
normally, which is why it is tested rather than reviewed.

**Rep counting.** Cross-checked two independent ways: the causal live counter and
the offline `find_peaks` segmenter. They agree on which clips contain reps,
including agreeing that `videoplayback (3).mp4` contains none across 98 seconds.

**Pose detection while supine.** 90–100% detection, core-joint visibility ≈0.999.
This was the largest identified risk in the original design document; it is
measured and it is not a problem. RTMPose was consequently dropped from the plan.

### What has *not* been evaluated

**Sensitivity to real errors.** No faulty footage exists. See
[LIMITATIONS.md](LIMITATIONS.md) §1. Until that is filmed, the system may be
described as measuring deviation from a reference, and may not be described as
detecting errors.

### Supporting benchmark (Track A)

The training and evaluation code is validated on published, labelled data that
does contain real errors — 39 Rehab-Pile datasets (KIMORE, KERAAL, IRDS,
KINECAL, SPHERE, UCDHE, UI-PRMD):

| model | mean rank | wins | mean macro-F1 | datasets |
|---|---|---|---|---|
| MiniRocket | **1.65** | 25 | 0.684 | 39 |
| RF (summary) | 2.17 | 11 | 0.610 | 39 |
| LITEMV | 2.67 | 1 | 0.582 | **3** |
| RF (flatten) | 2.76 | 6 | 0.528 | 39 |
| majority | 3.60 | 4 | 0.379 | 39 |

Reported as **mean rank over datasets**, because test folds hold 6–14 samples and
a raw-score average would let one easy dataset dominate. `class_weight='balanced'`
on RF and MiniRocket; aeon exposes no equivalent for LITEMV, so it runs
unweighted. LITEMV covers 3 datasets, not 39 — one dataset at 5 classifiers cost
7365 s and then died on memory. These are **library defaults, not the paper's
configuration**, which could not be recovered.

Track A shares `modeling/evaluate.py` with Track B — deliberately numpy-only so
both environments can import it. Same macro-F1, same LOSO splitter, same leakage
assertion.

## Features

`D = 109` per frame, COCO-17 layout:

| stream | channels | |
|---|---|---|
| joints | 34 | 17 × (x, y), normalized |
| bones | 32 | 16 × child − parent |
| velocity | 34 | 17 × first difference × fps |
| angles | 8 | unsigned, radians |
| lumbar | 1 | the silhouette signal |

COCO-17 rather than MediaPipe-33: the 16 extra landmarks are face and hand detail
that would add 32 noise channels to an exercise about the trunk. Signals still
index mp33 directly, because `z` (pelvic rotation) and `visibility` (the QC gate)
do not survive the projection.

## Data

See [DATASHEET.md](DATASHEET.md). In short: public YouTube footage plus the
author's own recordings, all `correct`, de-duplicated with an auditable audit
trail, `person_id` assigned by hand.

## Ethical considerations

- **No patient data.** No clinical population, no medical records, no consent
  regime beyond the author's own footage and publicly posted instructional video.
- **The demographic range is not characterised** and is certainly narrow. Body
  composition and clothing both affect a silhouette-based signal; there is no
  evidence here about how performance varies across body types, and none should
  be assumed.
- **Failure mode with a bad baseline.** The live app calibrates on the user's own
  first three reps. If those are performed badly, the app will approve the same
  fault for the rest of the session. A user could be reinforced in an error.
- **The system should not be presented as safety-critical feedback.** Someone
  exercising with back pain needs a clinician, not a control chart.

## Reproducing

```bash
uv venv venv --python 3.13 && uv pip install --python venv -r requirements.txt
./venv/Scripts/python.exe -m pytest tests/ -q          # 113 tests
```

Track A needs a separate environment — `aeon` pins `numpy<2.5`, `pandas<2.4`,
`scipy<1.18`, all below what the MediaPipe stack requires:

```bash
uv venv venv-a --python 3.13 && uv pip install --python venv-a -r requirements-a.txt
./venv-a/Scripts/python.exe scripts/run_masar_a.py --all --cheap
```

Every constant lives in `configs/base.yaml`; a run is described by its config
hash, recorded in the stage manifest alongside the git SHA.

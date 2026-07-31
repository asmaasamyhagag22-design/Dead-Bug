# Dead Bug AQA

Skeleton- and silhouette-based **Action Quality Assessment** for the Dead Bug core-stability exercise.

---

## The problem this solves

The primary error in a Dead Bug is the lower back arching off the floor. That error is
**invisible in the skeleton**: MediaPipe returns 33 landmarks and **not one of them is on
the spine**. A system built on keypoints alone cannot see the thing the exercise is about.

So the signal is read off the **segmentation mask** instead — the area enclosed between the
body's lower boundary and the floor line, over a window spanning the lumbar region:

```
lumbar_gap(t) = Σ max(0, floor_y(x) − bottom(x)) ÷ torso_len_px²
                 for x ∈ [0.00, 0.35] × torso_len along the hip→shoulder direction
```

Dividing by `torso_len²` — an area over a length squared — makes the signal dimensionless
and independent of camera distance and body size.

## Why it is built as two tracks

Every available Dead Bug clip is labelled `correct`. Training a multi-class error
classifier on them would be circular, so the project separates two questions that are
routinely conflated:

| | question | needs faulty data? |
|---|---|---|
| **Instrument validity** | does `lumbar_gap` really measure back height? | ❌ provable from correct-only footage |
| **Detector evaluation** | does it catch real errors? | ✅ no substitute exists |

- **Track A** — Rehab-Pile / KIMORE / KERAAL: published, labelled, contains real errors.
  Proves the training and evaluation code is correct and yields a defensible numbers table.
- **Track B** — Dead Bug: the original contribution. Silhouette lumbar signal, LOSO
  normative band, contralateral rep segmentation.

The two share `modeling/evaluate.py`, which is deliberately **numpy-only** so both
environments can import it: same macro-F1, same LOSO splitter, same leakage assertion.

## Status

| | |
|---|---|
| **Gate 0** — translation / scale / rotation / flip invariance | ✅ green |
| **V1 + V3** instrument validity, as unit tests with closed-form expected values | ✅ passing |
| Pose extraction, floor estimation, lumbar signal on real clips | ✅ working |
| Live coaching app — causal counting, per-session calibration, voice, report | ✅ working |
| Dataset / QC / normative-band / feature layers | ✅ built, 138 tests green |
| Track A — 39 Rehab-Pile datasets | ✅ done: MiniRocket 1st on 25/39, mean rank 1.65 |
| **Gate 1** — signal separates correct from arched reps | ⏳ blocked: no faulty footage exists yet |

The one remaining blocker is footage. Four faulted sessions (arched / fast /
rotated / correct) plus three short clips at known 0° / 30° / 45° would unblock
Gate 1, the C7 sensitivity check, and the `view_score` calibration all at once.
See [LIMITATIONS.md](LIMITATIONS.md) §1.

## Commands

```bash
deadbug scan  <video>     # does this file contain any exercise at all?
deadbug try   <url|path>  # fetch, scan, then coach -- the demo path
deadbug build             # the reps table
deadbug qc                # reports/qc.csv + qc.html
deadbug band              # the leave-one-subject-out normative band
```

Run them as `./venv/Scripts/python.exe -m deadbug.cli <command>`, or see the
`Makefile`.

## Quickstart

```bash
python -m venv venv && ./venv/Scripts/python.exe -m pip install -r requirements.txt
./venv/Scripts/python.exe -m pytest tests/ -v
```

Track A needs its own environment — `aeon` pins `numpy<2.5`, `pandas<2.4`, `scipy<1.18`
below what the MediaPipe stack uses:

```bash
uv venv venv-a --python 3.13
uv pip install --python venv-a "aeon==1.5.0" tensorflow keras matplotlib pyarrow pyyaml
./venv-a/Scripts/python.exe scripts/run_masar_a.py --sanity
```

⚠️ Pin `aeon==1.5.0` explicitly. A constraint like `numpy<2.5` lets the resolver settle on
`aeon==0.0.0` — an empty placeholder on PyPI — and the install **succeeds with exit code 0**.

## Limitations

> This system detects deviations from a reference movement pattern. It is not a diagnostic
> or clinical tool, was not validated on a patient population, and must not be used to
> guide medical decisions.

Small subject pool; any false-alarm rate is reported with a Wilson confidence interval and
with N stated in the sentence itself. Sensitivity to real errors is not yet evaluated.
See [PROCESS.md](PROCESS.md) §11.

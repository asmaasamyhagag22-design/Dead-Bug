# Filming protocol

The one blocker. Everything else in the project is built and tested; this is the
data that does not exist yet.

Read [LIMITATIONS.md](LIMITATIONS.md) §1 for why it cannot be worked around. In
short: every clip currently held is a `correct` performance, so the system can
be shown to *measure* back height and cannot be shown to *detect* an arch.

---

## Setup — five minutes, once

| | |
|---|---|
| **Camera** | Phone, on the floor, lens roughly at hip height |
| **Distance** | ~2 m to the side, whole body in frame with room to spare |
| **Angle** | Perpendicular to your body — a true side view |
| **Orientation** | Landscape |
| **Resolution** | 720p is enough. 1080p costs extraction time for no gain |
| **Floor** | Visible along the bottom of the frame, uncluttered |
| **Clothing** | Fitted. A loose t-shirt hangs and the silhouette follows the garment, not your spine — the pipeline cannot tell those apart |
| **Lighting** | Even. Avoid a bright window behind you; a backlit subject wrecks the mask |

**Before each take:** lie still on your back for **3 seconds** before the first
rep. That is the floor calibration, and it is the one thing the app gets that a
YouTube clip never provides. Knees at about 90°.

Do not trim, rotate, or re-export anything afterwards. Sub-clipping is a seek
inside the pipeline; a re-encode only loses quality and confuses the dedup pass.

---

## The four takes

Ten reps each, alternating sides. About one minute per take.

| # | condition | what to do | file name |
|---|---|---|---|
| 1 | `correct` | Lower back pressed to the floor throughout | `correct_01.mp4` |
| 2 | `arched` | **Deliberately let the lower back lift** on every rep | `arched_01.mp4` |
| 3 | `fast` | Correct form, but no pause — run the reps together | `fast_01.mp4` |
| 4 | `rotated` | Let the pelvis roll toward the extending leg | `rotated_01.mp4` |

Shoot **1 last**, not first. By the fourth take you know the movement, and the
`correct` reps are the ones the normative band is built from.

> The label comes from **your intent while filming**, which is a source
> independent of the signal. That is what makes the evaluation non-circular, and
> it is why `schema.py` refuses any manifest whose `label_source` is not
> `intent`. Do not go back afterwards and re-label a rep because the number
> looked wrong.

Exaggerate the faults. A barely-arched rep in take 2 that the system misses
tells you nothing — you cannot tell whether the instrument failed or the fault
was not really there. Make take 2 unmistakably wrong, establish that it is
caught, and only then ask how small a fault it can see.

## Plus three short angle clips

Twenty seconds each, **correct form**, camera moved to a known angle:

| angle | file name |
|---|---|
| 0° (true side) | `angle_00.mp4` |
| 30° | `angle_30.mp4` |
| 45° | `angle_45.mp4` |

These calibrate `triage.view_score_side_max` (currently 0.12) and
`view_score_oblique_max` (0.21), which today come from a geometric table rather
than from measurement. Mark the angles on the floor with tape before you start.

## Optional, if a second person is available

One `correct` take from someone else doubles the subject count. The
leave-one-subject-out band is fitted on everyone except the person being judged,
so at N=1 there is nothing to fit. **N=2 is the minimum at which the band means
anything at all**, and every additional person improves it more than another take
from the same one.

---

## After filming

```bash
# 1. copy the files in
cp /path/from/phone/*.mp4 data/clips/

# 2. measure everything and draft the manifest
python scripts/run_triage.py --write-manifest

# 3. edit data/clips.csv by hand -- person_id and condition
#    (the draft leaves person_id provisional; the schema will not validate
#     until it is real)

# 4. the rest
python -m deadbug.cli build
python -m deadbug.cli qc
python -m deadbug.cli band
```

Step 3 is the only manual one, and it stays manual on purpose: `person_id` is an
identity claim and `condition` is a statement about intent. Neither is in the
pixels, and a wrong `person_id` puts the same subject on both sides of a LOSO
split without anything downstream noticing.

### What to check before believing anything

```bash
python -m deadbug.cli qc          # then open reports/qc.html
```

- **`detection_rate` ≥ 0.90.** Below that, re-shoot with better lighting.
- **`view_score` < 0.12** on the four main takes, or they are not side views.
- **`torso_len_cv` ≤ 0.10.** Higher means the camera moved — put the phone down
  and do not hold it.
- **`n_reps` = 10** per take. If the segmenter finds 8, watch the preview in
  `reports/preview/` before assuming the segmenter is wrong.
- **`floor inliers` ≥ 0.80**, or the floor line is not where you think it is and
  every lumbar number from that clip is arbitrary.

### The result that would settle Gate 1

A single figure: `lumbar_gap` against excursion, `correct` reps in one colour
and `arched` in another, with the LOSO band drawn behind them. If the two clouds
separate, the instrument detects the error. If they overlap, it does not — and
that is a real finding, reportable as such, not a failure of the project.

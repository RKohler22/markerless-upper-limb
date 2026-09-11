# Markerless upper-limb kinematics — an error propagation study

How measurement error propagates through a multi-camera markerless pipeline,
and what that implies for quality assurance of kinematic datasets.

The study has three parts. Part 1 works on synthetic data with known ground
truth, isolating individual error mechanisms. Part 2 tests whether those
mechanisms appear in real recordings, using the released results of the
OpenCap laboratory validation dataset. Part 3 implements the pipeline
independently from the raw videos of the same dataset, in order to separate
reconstruction error from the joint-definition step that follows it.

The design is deliberate. On real recordings the true joint angle is never
available, so the accuracy of a reconstruction cannot be separated from the
accuracy of its reference — the marker-based system is itself subject to
soft-tissue artefact. Only a synthetic scene makes the truth accessible.

---

## Part 1 — Synthetic study

Pipeline: camera calibration → distortion removal → 2D keypoints in two views
→ DLT triangulation → 3D joint angle. Each stage is verified against a known
value before the next is added (`test_geometry.py`). Without noise the full
chain reproduces ground truth to machine precision (3.6e-12 m, 2.0e-10°).

### Finding 1 — reprojection error does not measure calibration validity

Intrinsics recovered from 20 synthetic checkerboard views at 0.3 px detection
noise, over 10 independent draws per condition.

| board inclination | mean \|focal error\| | SD | max | RMS reprojection |
|---|---|---|---|---|
| 0.03 rad | 5.85 % | 3.69 | 10.83 % | 0.411 px |
| 0.10 rad | 1.07 % | 0.62 | 1.96 % | 0.411 px |
| 0.20 rad | 0.30 % | 0.21 | 0.65 % | 0.411 px |
| 0.40 rad | 0.13 % | 0.09 | 0.27 % | 0.411 px |
| 0.60 rad | 0.13 % | 0.08 | 0.32 % | 0.411 px |

The RMS reprojection error is identical to three decimals across all
conditions while the focal length error varies by a factor of 45. With poor
inclination the calibration is not merely inaccurate but unreliable (SD is
63 % of the mean), so the magnitude of the error is itself unknown.

**Implication.** Reprojection error quantifies the internal consistency of the
fit, not the validity of the parameters. Parameter dispersion across repeated
calibrations — bootstrapped over subsets of views — is proposed as the quality
criterion instead. Inclination beyond a median of roughly 23° yields no
further benefit.

Radial coefficients are correlated: k2 and k3 deviate in opposite directions
while their combined effect over the observed radial range remains correct.
Individual distortion coefficients should not be interpreted in isolation.

### Finding 2 — angular error propagates linearly

With two cameras at 1.8 m and a 60° inter-camera angle, angular error scales
at **0.52° per pixel** of Gaussian keypoint noise (0.52 / 1.56 / 2.60° RMSE at
σ = 1 / 3 / 5 px). A single coefficient therefore transfers to any keypoint
detector whose localisation error is known, without repeating the simulation.

### Finding 3 — error is not uniform across the range of motion

Near full extension the three points approach collinearity and both
perpendicular noise components contribute to first order, rather than one.

| | RMSE (σ = 3 px) | bias |
|---|---|---|
| mid-range (70–110°) | 1.34° | −0.001° |
| near extension (>160°) | 1.97° | −0.267° |

The RMSE ratio of 1.47 is close to the √2 predicted by the collinearity
argument; the excess is attributable to rig anisotropy. The bias arises
because 180° bounds the achievable angle, so noise can only reduce the
measured value.

### Finding 4 — bias and precision respond oppositely to rig geometry

Rotating the plane of motion from transverse to aligned with the stereo depth
direction (σ = 3 px, 60° rig):

| plane azimuth | overall RMSE | RMSE near extension | bias near extension |
|---|---|---|---|
| 0° | 1.32° | 1.58° | −0.270° |
| 45° | 1.66° | 2.17° | −0.199° |
| 90° | 1.97° | 2.59° | −0.124° |

RMSE rises by a factor of 1.49 while bias falls by roughly half. The two
mechanisms are decoupled: overall error is dominated by depth uncertainty,
whereas the collinearity bias depends only on the noise component lying in
the plane of the limb.

**Implication.** No rig orientation minimises both. A single aggregate figure
such as "RMSE below X°" is insufficient as an acceptance criterion; bias and
precision must be reported separately, and both are specific to the movement
being measured rather than to the camera setup alone. A rig configured for
gait may be poorly conditioned for reaching movements without any property of
the setup indicating it.

---

## Part 2 — Released results (OpenCap laboratory validation set)

Ten subjects, five calibrated smartphone cameras, marker-based reference
kinematics, and markerless results precomputed for three keypoint detectors
and 2/3/5 cameras. Reference and markerless output share an identical time
base, so frames are compared directly; trials failing that check are excluded
rather than interpolated.

Note on convention: OpenSim reports `knee_angle` and `elbow_flex` as flexion
from zero, whereas the synthetic model uses 180° for full extension. Bias
signs are mirrored between the parts accordingly.

### Finding 5 — lower-limb validation figures do not transfer to the upper limb

28 walking trials, 10 subjects, HRNet, 2 cameras. Values are mean ± SD across
trials.

| joint | RMSE | bias | scatter | reference ROM |
|---|---|---|---|---|
| knee | 4.44 ± 1.67° | +1.25 ± 2.96° | 3.35 ± 1.10° | 68° |
| elbow | 15.09 ± 3.88° | −14.72 ± 3.96° | 3.10 ± 0.93° | 31° |

Knee bias is near zero at cohort level while individual subjects deviate
clearly, indicating subject-specific offsets that average out across a group
but remain unknown for any single patient — the situation in clinical
assessment.

Elbow error is almost entirely systematic: the bias is roughly five times the
residual scatter, and comparable to the joint's range of motion during gait.
The published OpenCap validation figure of approximately 4.5° refers to
lower-limb kinematics; the knee result here is consistent with it, the elbow
result is not.

### Finding 6 — camera count does not help

| joint | 2 cameras | 3 cameras | 5 cameras |
|---|---|---|---|
| knee, bias | +1.25 ± 2.96° | +0.61 ± 3.30° | −0.46 ± 2.87° |
| knee, scatter | 3.35 ± 1.10° | 4.50 ± 1.57° | 4.27 ± 1.56° |
| elbow, bias | −14.72 ± 3.96° | −14.72 ± 4.05° | −14.20 ± 4.18° |
| elbow, scatter | 3.10 ± 0.93° | 3.16 ± 1.03° | 3.00 ± 1.03° |

Neither joint improves. All differences between 2, 3 and 5 cameras lie within
between-subject variability. For the elbow this is consistent with a
joint-definition rather than a reconstruction-geometry limitation: added
viewpoints cannot correct a definition.

### Finding 7 — elbow error is task specific

HRNet, 2 cameras, all subjects:

| task | reference ROM | bias | scatter | n |
|---|---|---|---|---|
| walking | 30.95 ± 11.82° | −14.72 ± 3.96° | 3.10 ± 0.93° | 28 |
| sit-to-stand | 23.71 ± 16.36° | −7.50 ± 3.37° | 5.02 ± 1.19° | 8 |
| squats | 51.68 ± 25.86° | −4.78 ± 5.35° | 5.53 ± 2.82° | 9 |
| drop jump | 67.76 ± 28.54° | −5.29 ± 6.17° | 11.04 ± 7.83° | 26 |

Gait separates clearly from all other tasks, and the separation exceeds
within-task variability. Bias does not scale monotonically with range of
motion — sit-to-stand has the smallest ROM of the non-gait tasks yet not the
largest bias — so amplitude alone does not explain the effect. Scatter moves
in the opposite direction to bias, mirroring the decoupling seen
synthetically in finding 4.

**Implication.** Error characteristics of markerless upper-limb kinematics are
task specific. A single validation figure per joint is inadequate for a
quality assurance framework: error must be characterised for the movement
class under study, and bias and precision reported separately.

---

## Part 3 — Independent reimplementation from the raw videos

Findings 5 to 7 measure the released pipeline as a whole. That pipeline
includes an LSTM step that converts video keypoints into anatomical marker
positions, so its error is the sum of keypoint localisation, triangulation
and augmentation. To separate them, the geometric part was reimplemented from
the raw videos of the same dataset: YOLO11-pose keypoints (COCO right
shoulder, elbow, wrist), two calibrated views, distortion removal, DLT
triangulation, three-point joint angle. No augmentation step.

Verification before any comparison: reconstructed points fall near the centre
of the camera arc, and reconstructed segment lengths are anatomically
plausible. Camera parameters are loaded per subject, since extrinsics are
session specific.

### Finding 8 — the augmentation step corrects bias, not precision

Eight squat and sit-to-stand trials from four subjects, cameras 0 and 2
(3.5 m baseline), compared against the marker-based reference on a common
60 Hz time base.

| | bias | scatter |
|---|---|---|
| own pipeline, raw keypoints | −20.80 ± 2.95° | 8.09 ± 3.24° |
| released OpenCap result | −5.37 ± 4.54° | 6.25 ± 2.44° |

Scatter is of comparable magnitude, so reconstruction geometry is not the
limiting factor. What the augmentation step contributes is a relocation of
the joint centre from the COCO keypoint to an anatomical position. The raw
bias is also the more consistent of the two — its SD is 14 % of its mean,
against 85 % for the augmented result — as expected for a definitional offset
rather than a measurement error.

### Finding 9 — segment length variation as a reference-free quality measure

The forearm and upper arm are rigid, so any variation in their reconstructed
length over time is reconstruction error made visible without a reference
system.

| segment | SD across frames |
|---|---|
| upper arm | 11.5 ± 3.2 mm |
| forearm | 18.3 ± 4.9 mm |

The distal segment is consistently worse, as expected: the wrist is the point
furthest from stable landmarks and the fastest moving. For a 225 mm forearm,
an 18 mm length variation corresponds to roughly 5° of angular uncertainty —
the same order as the released pipeline's error on these tasks.

**Implication.** This is a candidate acceptance criterion for clinical
settings, where no marker-based reference is available: it needs only the
reconstruction itself and an assumption that limbs are rigid.

---

## Limitations

- Part 3 separates augmentation from reconstruction for squats and
  sit-to-stand only. The video release does not contain the walking trials,
  which is where the largest elbow bias was observed (finding 5), so the
  separation cannot be performed for the task where it matters most.
- Arm motion during gait is incidental rather than a target movement. Whether
  the offsets persist during reaching remains open; no task in this dataset
  is an upper-limb reaching task.
- Parts 2 and 3 use one detector each (HRNet and YOLO11-pose respectively) and
  a single camera pair in part 3; the absolute figures are specific to those
  choices.
- The marker-based reference is itself subject to soft-tissue artefact, so all
  figures in parts 2 and 3 are agreement measures, not accuracy measures.
- Part 1 models a two-segment planar chain; real joints have more degrees of
  freedom and additional error sources.
- The released intrinsics are deployed per phone model rather than estimated
  per session, so manufacturing variation between devices enters every
  reconstruction unchecked.

## Files

| file | contents |
|---|---|
| `geometry.py` | verified geometric primitives; conventions in the header |
| `synthetic.py` | synthetic scenes with known ground truth |
| `io_opensim.py` | OpenSim `.mot`/`.sto` reading, with header verification |
| `test_geometry.py` | verification suite — run before working with real data |
| `01_synthetic_error_study.py` | Part 1, findings 1–4 |
| `02_opencap_exploration.py` | Part 2, findings 5–7 |
| `03_video_pipeline.py` | Part 3, findings 8–9 |

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe test_geometry.py     # must report 5/5
```

Results were verified identically under OpenCV 4.11 and 5.0.

Part 1 runs standalone. Part 2 expects the OpenCap laboratory validation set
under `data/LabValidation_withoutVideos/` and part 3 the video release under
`data/LabValidation_withVideos/`, both available from SimTK after
registration. Part 3 additionally requires `ultralytics`. The data directory
is excluded from version control: the set contains identifiable video and is
subject to a data use agreement.

## Third-party material

This work analyses the OpenCap laboratory validation dataset, released by
Stanford University under the Apache License 2.0. **No Stanford code is
redistributed here; all analysis code in this repository is original.**
Dataset and reference implementation: https://simtk.org/projects/opencap

> Copyright (c) 2022 Stanford University. Licensed under the Apache License,
> Version 2.0 (the "License"); you may not use this file except in compliance
> with the License. You may obtain a copy of the License at
> http://www.apache.org/licenses/LICENSE-2.0 — Unless required by applicable
> law or agreed to in writing, software distributed under the License is
> distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
> KIND, either express or implied.

## License

Analysis code in this repository: Apache License 2.0, Copyright 2026 Roger
Kohler. See `LICENSE`.

## Reference

Uhlrich, S. D. et al. (2023). OpenCap: Human movement dynamics from
smartphone videos. *PLOS Computational Biology*.

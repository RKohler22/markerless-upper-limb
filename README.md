# Markerless upper-limb kinematics — an error propagation study

How measurement error propagates through a multi-camera markerless pipeline,
and what that implies for quality assurance of kinematic datasets.

The study has two parts. Part 1 works on synthetic data with known ground
truth, isolating individual error mechanisms. Part 2 tests whether those
mechanisms are visible in real recordings, using the OpenCap laboratory
validation dataset.

The two-part design is deliberate. On real recordings the true joint angle is
never available, so the accuracy of a reconstruction cannot be separated from
the accuracy of its reference — the marker-based system is itself subject to
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

## Part 2 — Real recordings (OpenCap laboratory validation set)

Ten subjects, five calibrated smartphone cameras, marker-based reference
kinematics, and markerless results precomputed for three keypoint detectors
and 2/3/5 cameras. Reference and markerless output share an identical time
base, so frames are compared directly; trials failing that check are excluded
rather than interpolated.

Note on convention: OpenSim reports `knee_angle` and `elbow_flex` as flexion
from zero, whereas the synthetic model uses 180° for full extension. Bias
signs are mirrored between the two parts accordingly.

### Finding 5 — lower-limb validation figures do not transfer to the upper limb

28 walking trials, 10 subjects, HRNet, 2 cameras. All values mean ± SD across
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

## Limitations

- Markerless outputs in this dataset pass through an LSTM marker-augmentation
  step. The measured error is therefore the sum of keypoint localisation,
  triangulation and augmentation error, and cannot be attributed to
  triangulation alone. Separating them would require computing keypoints and
  triangulating them independently from the video release.
- Arm motion during gait is incidental rather than a target movement. Whether
  the offset persists during reaching remains open; no task in this dataset
  is an upper-limb reaching task.
- The task comparison uses one detector (HRNet) and two cameras.
- The marker-based reference is itself subject to soft-tissue artefact, so all
  figures in Part 2 are agreement measures, not accuracy measures.
- Part 1 models a two-segment planar chain; real joints have more degrees of
  freedom and additional error sources.

## Files

| file | contents |
|---|---|
| `geometry.py` | verified geometric primitives; conventions in the header |
| `synthetic.py` | synthetic scenes with known ground truth |
| `io_opensim.py` | OpenSim `.mot`/`.sto` reading, with header verification |
| `test_geometry.py` | verification suite — run before working with real data |
| `01_synthetic_error_study.py` | Part 1, findings 1–4 |
| `02_opencap_exploration.py` | Part 2, findings 5–7 |

## Reproducing

```bash
python -m pip install -r requirements.txt
python test_geometry.py          # must report 5/5
```

Part 1 runs standalone. Part 2 expects the OpenCap laboratory validation set
under `data/LabValidation_withoutVideos/`, available from SimTK after
registration. The data directory is excluded from version control: the set
contains identifiable video and is subject to a data use agreement.

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

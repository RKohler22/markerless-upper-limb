# Markerless upper-limb kinematics — validation study

Small self-contained study on how measurement error propagates through a
two-camera markerless pipeline, and what that means for quality assurance
of kinematic data.

All results below are obtained on synthetic data with known ground truth.
This is deliberate: with real recordings the true joint angle is never
available, so the accuracy of a reconstruction cannot be separated from the
accuracy of its reference.

## Pipeline

Camera calibration → distortion removal → 2D keypoints in two views →
DLT triangulation → 3D joint angle.

Every stage is verified against a known value before the next is added
(`test_geometry.py`). Without noise the full chain reproduces ground truth
to machine precision (3.6e-12 m, 2.0e-10°).

## Finding 1 — reprojection error does not measure calibration validity

Intrinsics were recovered from 20 synthetic checkerboard views with 0.3 px
detection noise, repeated over 10 independent draws per condition.

| board inclination | mean \|focal error\| | SD | max | RMS reprojection |
|---|---|---|---|---|
| 0.03 rad | 5.85 % | 3.69 | 10.83 % | 0.411 px |
| 0.10 rad | 1.07 % | 0.62 | 1.96 % | 0.411 px |
| 0.20 rad | 0.30 % | 0.21 | 0.65 % | 0.411 px |
| 0.40 rad | 0.13 % | 0.09 | 0.27 % | 0.411 px |
| 0.60 rad | 0.13 % | 0.08 | 0.32 % | 0.411 px |

The RMS reprojection error is identical to three decimals across all
conditions while the focal length error varies by a factor of 45. With poor
board inclination the calibration is not merely inaccurate but unreliable
(SD is 63 % of the mean), so the magnitude of the error is itself unknown.

**Implication.** Reprojection error quantifies the internal consistency of
the fit, not the validity of the parameters. Parameter dispersion across
repeated calibrations — bootstrapped over subsets of views — is proposed as
the quality criterion instead. Inclination beyond a median of roughly 23°
yields no further benefit.

Radial coefficients are correlated: k2 and k3 deviate in opposite directions
while their combined effect over the observed radial range remains correct.
Individual distortion coefficients should not be interpreted in isolation.

## Finding 2 — angular error propagates linearly

With two cameras at 1.8 m and a 60° inter-camera angle, angular error scales
at **0.52° per pixel** of Gaussian keypoint noise (0.52 / 1.56 / 2.60° RMSE
at σ = 1 / 3 / 5 px). A single coefficient therefore transfers to any
keypoint detector once its localisation error is known, without repeating
the simulation.

## Finding 3 — error is not uniform across the range of motion

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

**Clinical relevance.** Extension deficit is a primary outcome in upper-limb
assessment after stroke. Precision is lowest and bias largest exactly where
the measurement matters, and the bias direction mimics the pathology.
Unlike random error, it does not average out over repeated trials, and it
changes as a patient's range of motion changes during rehabilitation.

## Finding 4 — bias and precision respond oppositely to rig geometry

Rotating the plane of motion from transverse to aligned with the stereo
depth direction (σ = 3 px, 60° rig):

| plane azimuth | overall RMSE | RMSE near extension | bias near extension |
|---|---|---|---|
| 0° | 1.32° | 1.58° | −0.270° |
| 45° | 1.66° | 2.17° | −0.199° |
| 90° | 1.97° | 2.59° | −0.124° |

RMSE rises by a factor of 1.49 while bias falls by roughly half. The two
mechanisms are decoupled: overall error is dominated by depth uncertainty,
whereas the collinearity bias depends only on the noise component lying in
the plane of the limb.

**Implication.** No rig orientation minimises both. A single aggregate
figure such as "RMSE below X°" is insufficient as an acceptance criterion;
bias and precision must be reported separately, and both are specific to the
movement being measured rather than to the camera setup alone. A rig
configured for gait may be poorly conditioned for reaching movements without
any property of the setup indicating it.

## Files

- `geometry.py` — verified building blocks; conventions documented in the header
- `test_geometry.py` — verification suite, run before working with real data

## Next

Apply the same pipeline to the OpenCap laboratory validation dataset
(Uhlrich et al., 2023), where synchronised marker-based reference kinematics
allow the synthetic error model to be checked against real recordings.

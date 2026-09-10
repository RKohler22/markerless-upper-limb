# -*- coding: utf-8 -*-
"""
01_synthetic_error_study.py

Error propagation in a two-camera markerless pipeline, studied on
synthetic data where ground truth is known exactly.

All reusable functions live in geometry.py (methods under test) and
synthetic.py (test material). This file only runs experiments on them.

Spyder:
    Ctrl+Enter    run cell
    Shift+Enter   run cell and advance
    Ctrl+.        restart kernel - do this first whenever state is unclear
"""

# %% Cell 0 - environment

import numpy as np
import matplotlib.pyplot as plt

from geometry import (make_board_points, calibrate, make_stereo_rig,
                      projection_matrix, project_points, undistort,
                      triangulate, joint_angle)
from synthetic import (make_ground_truth_camera, project_pose, make_pose_set,
                       add_detection_noise, make_arm)

print("environment ready")


# %% Cell 1 - single board pose, visual and numeric check

def plot_projection(uv, size, title="projection"):
    """Visual check: are the corners inside the image and is the grid
    plausibly distorted?"""
    w, h = size
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(uv[:, 0], uv[:, 1], "o", ms=3)
    ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False, lw=1.5))
    ax.set_xlim(-100, w + 100)
    ax.set_ylim(h + 100, -100)          # image coordinates: v points down
    ax.set_aspect("equal")
    ax.set_xlabel("u [px]"); ax.set_ylabel("v [px]"); ax.set_title(title)
    plt.tight_layout(); plt.show()


board = make_board_points()
K, dist, size = make_ground_truth_camera()

uv = project_pose(board, K, dist,
                  rvec=[0.30, -0.40, 0.05],      # inclined
                  tvec=[-0.12, -0.075, 0.60])    # 60 cm in front of camera

print("board points :", board.shape)
print("image points :", uv.shape)
print("min [px]     :", uv.min(0).round(1))
print("max [px]     :", uv.max(0).round(1))
print("width  [px]  :", round(np.ptp(uv[:, 0]), 1))
print("height [px]  :", round(np.ptp(uv[:, 1]), 1))

plot_projection(uv, size, "one inclined board pose")

# Expected: 54 points, all inside 0..1920 / 0..1080, board 400-700 px wide.
# Cross-check: 0.24 m * 1400 px / 0.60 m = 560 px if seen frontally; the
# measured width is lower because of inclination and barrel distortion.


# %% Cell 2 - pose set: 20 valid views

poses, views, tries = make_pose_set(board, K, dist, size, n=20, seed=0)

tilts = np.degrees([np.linalg.norm(rv) for rv, _ in poses])
alluv = np.vstack(views)

print("poses        :", len(poses), "from", tries, "attempts")
print("tilt angle   : min %.1f  median %.1f  max %.1f deg"
      % (tilts.min(), np.median(tilts), tilts.max()))
print("coverage u   : %.0f .. %.0f of %d" % (alluv[:, 0].min(),
                                             alluv[:, 0].max(), size[0]))
print("coverage v   : %.0f .. %.0f of %d" % (alluv[:, 1].min(),
                                             alluv[:, 1].max(), size[1]))

fig, ax = plt.subplots(figsize=(8, 5))
for uv_i in views:
    ax.plot(uv_i[:, 0], uv_i[:, 1], ".", ms=2)
ax.add_patch(plt.Rectangle((0, 0), size[0], size[1], fill=False, lw=1.5))
ax.set_xlim(-50, size[0] + 50); ax.set_ylim(size[1] + 50, -50)
ax.set_aspect("equal"); ax.set_title("coverage of all 20 poses")
plt.tight_layout(); plt.show()

# Corners far from the principal point carry the distortion information.
# Poor edge coverage leaves k2 and k3 extrapolated rather than estimated.


# %% Cell 3 - calibration from noisy detections, compared to truth

def compare_to_truth(K_est, dist_est, K, dist):
    """The point of the synthetic approach: the truth is available."""
    names = ["fx", "fy", "cx", "cy"]
    true = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    est = [K_est[0, 0], K_est[1, 1], K_est[0, 2], K_est[1, 2]]
    print(f"{'param':6} {'true':>10} {'estimated':>12} {'diff':>10} {'diff %':>8}")
    for n, t, e in zip(names, true, est):
        print(f"{n:6} {t:10.2f} {e:12.2f} {e-t:10.2f} {100*(e-t)/t:8.2f}")
    for i, n in enumerate(["k1", "k2", "p1", "p2", "k3"]):
        print(f"{n:6} {dist[i]:10.4f} {dist_est[i]:12.4f} "
              f"{dist_est[i]-dist[i]:10.4f}")


SIGMA = 0.3
views_noisy = add_detection_noise(views, sigma_px=SIGMA)
rms, K_est, dist_est = calibrate(board, views_noisy, size)

print(f"injected noise    : {SIGMA:.2f} px")
print(f"RMS reprojection  : {rms:.3f} px\n")
compare_to_truth(K_est, dist_est, K, dist)

# The RMS should land close to the injected noise. Substantially lower
# would indicate overfitting; substantially higher, a correspondence error.


# %% Cell 4 - FINDING 1: reprojection error does not measure validity

def run_experiment(tilt_max, n=20, sigma=0.3, seed=0):
    """One full calibration with a prescribed inclination range."""
    board = make_board_points()
    K, dist, size = make_ground_truth_camera()
    poses, views, _ = make_pose_set(board, K, dist, size,
                                    n=n, seed=seed, tilt_max=tilt_max)
    if len(poses) < n:
        return None
    noisy = add_detection_noise(views, sigma_px=sigma, seed=seed + 1)
    rms, K_est, dist_est = calibrate(board, noisy, size)
    return {"tilt_max": tilt_max, "n": len(poses), "rms": rms,
            "fx": K_est[0, 0], "fx_err_pct": 100 * (K_est[0, 0] - 1400) / 1400,
            "k1": dist_est[0]}


def sweep(tilt_values, seeds=range(10)):
    """Repeat each condition over independent random draws.

    A single draw is an anecdote: the mean separates the systematic part,
    the standard deviation the random part. That distinction is the core
    of any quality assurance statement.
    """
    rows = []
    for t in tilt_values:
        errs, rmss = [], []
        for s in seeds:
            r = run_experiment(t, seed=s)
            if r is not None:
                errs.append(abs(r["fx_err_pct"]))
                rmss.append(r["rms"])
        rows.append({"tilt_max": t, "n": len(errs),
                     "err_mean": np.mean(errs), "err_std": np.std(errs),
                     "err_max": np.max(errs), "rms_mean": np.mean(rmss)})
    return rows


rows = sweep([0.03, 0.10, 0.20, 0.40, 0.60])

print(f"{'tilt_max':>9} {'n':>3} {'|fx err| %':>12} {'SD':>7} {'max':>7} {'RMS [px]':>9}")
for r in rows:
    print(f"{r['tilt_max']:9.2f} {r['n']:3d} {r['err_mean']:12.2f} "
          f"{r['err_std']:7.2f} {r['err_max']:7.2f} {r['rms_mean']:9.3f}")

# Result: RMS is constant to three decimals while the focal error varies
# by a factor of 45. Reprojection error quantifies internal consistency
# of the fit, not the validity of the parameters.


# %% Cell 5 - stereo rig and arm projection

arm, theta_true = make_arm()

theta_check = joint_angle(arm[:, 0], arm[:, 1], arm[:, 2])
print("shape        :", arm.shape)
print("angle range  : %.1f .. %.1f deg" % (theta_true.min(), theta_true.max()))
print("round trip   : max deviation %.2e deg"
      % np.abs(theta_check - theta_true).max())

target = arm.reshape(-1, 3).mean(axis=0)
cams = make_stereo_rig(target, radius=1.8, angle_deg=60.0)

for i, cam in enumerate(cams):
    C_rec = -cam["R"].T @ cam["t"]
    print(f"camera {i}: C = {np.round(cam['C'], 3)}   "
          f"round trip {np.abs(C_rec - cam['C']).max():.2e}")

views_arm = []
for i, cam in enumerate(cams):
    uv_i = project_points(arm.reshape(-1, 3), cam, K, dist).reshape(len(arm), 3, 2)
    views_arm.append(uv_i)
    f = uv_i.reshape(-1, 2)
    ok = ((f[:, 0] > 0) & (f[:, 0] < size[0]) &
          (f[:, 1] > 0) & (f[:, 1] < size[1])).all()
    print(f"camera {i}: wrist u-range {np.ptp(uv_i[:, 2, 0]):5.0f} px  "
          f"v-range {np.ptp(uv_i[:, 2, 1]):5.0f} px  all inside: {ok}")

# The two u-ranges must differ. If they were equal, the plane of motion
# would be symmetric to the rig and triangulation would not be tested in
# the depth direction, which is exactly the poorly conditioned one.


# %% Cell 6 - FINDING 2: angular error scales linearly with keypoint noise

def reconstruct_arm(views_arm, cams, K, dist, sigma_px=0.0, seed=0):
    """Full chain: noise -> undistort -> triangulate -> joint angle."""
    rng = np.random.default_rng(seed)
    n = views_arm[0].shape[0]
    P = [projection_matrix(c, K) for c in cams]
    uv = [v.reshape(-1, 2) + rng.normal(0, sigma_px, (n * 3, 2))
          for v in views_arm]
    uvu = [undistort(u, K, dist) for u in uv]
    X = triangulate(uvu[0], uvu[1], P[0], P[1]).reshape(n, 3, 3)
    return X, joint_angle(X[:, 0], X[:, 1], X[:, 2])


X0, theta0 = reconstruct_arm(views_arm, cams, K, dist, sigma_px=0.0)
print("WITHOUT NOISE")
print("  max point error : %.2e m" % np.abs(X0 - arm).max())
print("  max angle error : %.2e deg" % np.abs(theta0 - theta_true).max())

for s in [1.0, 3.0, 5.0]:
    e = np.array([reconstruct_arm(views_arm, cams, K, dist,
                                  sigma_px=s, seed=seed)[1] - theta_true
                  for seed in range(20)])
    print(f"\nsigma = {s:.1f} px")
    print(f"  bias        : {e.mean():+.2f} deg")
    print(f"  RMSE        : {np.sqrt((e**2).mean()):.2f} deg")
    print(f"  max |error| : {np.abs(e).max():.2f} deg")

# Result: 0.52 deg RMSE per pixel of keypoint noise for this geometry.
# A single transferable coefficient - no need to repeat the simulation
# for a detector whose localisation error is known.


# %% Cell 7 - FINDING 3: error is not uniform across the range of motion

def error_profile(views_arm, cams, K, dist, theta_true,
                  sigma_px=3.0, n_seeds=100):
    e = np.array([reconstruct_arm(views_arm, cams, K, dist,
                                  sigma_px=sigma_px, seed=s)[1] - theta_true
                  for s in range(n_seeds)])
    return e.mean(0), np.sqrt((e ** 2).mean(0))


bias_f, rmse_f = error_profile(views_arm, cams, K, dist, theta_true)

print(f"RMSE over frames: min {rmse_f.min():.2f}  median "
      f"{np.median(rmse_f):.2f}  max {rmse_f.max():.2f} deg")
print(f"worst frame at theta = {theta_true[rmse_f.argmax()]:.0f} deg")
print(f"best  frame at theta = {theta_true[rmse_f.argmin()]:.0f} deg")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(theta_true, rmse_f, lw=1.5)
ax.set_xlabel("true elbow angle [deg]"); ax.set_ylabel("RMSE [deg]")
ax.set_title("error profile at sigma = 3 px"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

i_ext = theta_true > 160
i_mid = (theta_true > 70) & (theta_true < 110)
print(f"bias near extension (>160 deg): {bias_f[i_ext].mean():+.3f} deg")
print(f"bias mid-range                : {bias_f[i_mid].mean():+.3f} deg")
print(f"RMSE ratio extension / mid    : "
      f"{rmse_f[i_ext].mean()/rmse_f[i_mid].mean():.2f}")

# Near extension the three points approach collinearity, so both
# perpendicular noise components contribute to first order instead of one.
# Predicted ratio sqrt(2) = 1.414, measured 1.47. The bias arises because
# 180 deg bounds the angle, so noise can only reduce it.


# %% Cell 8 - FINDING 4: bias and precision respond oppositely to geometry

def eval_config(plane_azimuth_deg, radius=1.8, rig_angle=60.0,
                sigma_px=3.0, n_seeds=50):
    """One full configuration: build arm, project, reconstruct, compare."""
    arm_c, theta_c = make_arm(plane_azimuth_deg=plane_azimuth_deg)
    target_c = arm_c.reshape(-1, 3).mean(axis=0)
    cams_c = make_stereo_rig(target_c, radius=radius, angle_deg=rig_angle)

    views_c = [project_points(arm_c.reshape(-1, 3), c, K, dist)
               .reshape(len(arm_c), 3, 2) for c in cams_c]

    flat = np.vstack([v.reshape(-1, 2) for v in views_c])
    inside = bool(((flat[:, 0] > 0) & (flat[:, 0] < size[0]) &
                   (flat[:, 1] > 0) & (flat[:, 1] < size[1])).all())

    e = np.array([reconstruct_arm(views_c, cams_c, K, dist,
                                  sigma_px=sigma_px, seed=s)[1] - theta_c
                  for s in range(n_seeds)])
    ext = theta_c > 160
    return {"az": plane_azimuth_deg, "inside": inside,
            "rmse": np.sqrt((e ** 2).mean()),
            "rmse_ext": np.sqrt((e[:, ext] ** 2).mean()),
            "bias_ext": e[:, ext].mean()}


res = [eval_config(a) for a in [0, 15, 30, 45, 60, 75, 90]]

print(f"{'azimuth':>8} {'inside':>7} {'RMSE':>8} {'RMSE ext':>9} {'bias ext':>9}")
for r in res:
    print(f"{r['az']:8.0f} {str(r['inside']):>7} {r['rmse']:8.2f} "
          f"{r['rmse_ext']:9.2f} {r['bias_ext']:+9.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([r["az"] for r in res], [r["rmse"] for r in res], "-o", label="overall")
ax.plot([r["az"] for r in res], [r["rmse_ext"] for r in res], "-s",
        label="near extension")
ax.set_xlabel("azimuth of the plane of motion [deg]")
ax.set_ylabel("RMSE [deg]")
ax.set_title("rig conditioning at sigma = 3 px")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# RMSE rises by a factor of 1.49 from 0 to 90 deg while the collinearity
# bias halves. The two mechanisms are decoupled: overall error is driven
# by depth uncertainty, the bias only by the noise component lying in the
# plane of the limb. No rig orientation minimises both.

# %% Cell 0 - environment

import sys, os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
    
import pickle
import numpy as np
import matplotlib.pyplot as plt

from geometry import (projection_matrix, undistort, triangulate,
                      joint_angle, segment_lengths)
from io_opensim import read_mot

import pandas as pd

print("environment ready")

# %% Block 3, Zelle 1 - Kameraparameter laden und pruefen

VID = Path("data/LabValidation_withVideos/subject2/VideoData/Session0")


def load_camera(cam_dir):
    """Laedt Intrinsics und Extrinsics einer OpenCap-Kamera.

    Konvertiert die Translation von Millimetern auf Meter, um die
    Konvention aus geometry.py einzuhalten.
    """
    with open(Path(cam_dir) / "cameraIntrinsicsExtrinsics.pickle", "rb") as f:
        d = pickle.load(f)
    R = np.asarray(d["rotation"], float)
    t_mm = np.asarray(d["translation"], float).ravel()
    return {
        "name": Path(cam_dir).name,
        "K": np.asarray(d["intrinsicMat"], float),
        "dist": np.asarray(d["distortion"], float).ravel(),
        "R": R,
        "t": t_mm / 1000.0,                 # mm -> m
        "C": -R.T @ (t_mm / 1000.0),
        "imageSize_declared": np.asarray(d["imageSize"], float).ravel(),
    }


cams = [load_camera(VID / f"Cam{i}") for i in range(5)]

print(f"{'cam':5} {'det(R)':>8} {'fx':>8} {'fy':>8} {'cx':>8} {'cy':>8} "
      f"{'|C| [m]':>9} {'C (x,y,z) [m]':>28}")
for c in cams:
    print(f"{c['name']:5} {np.linalg.det(c['R']):8.5f} "
          f"{c['K'][0,0]:8.1f} {c['K'][1,1]:8.1f} "
          f"{c['K'][0,2]:8.1f} {c['K'][1,2]:8.1f} "
          f"{np.linalg.norm(c['C']):9.2f}   {np.round(c['C'], 2)}")

print("\npaarweise Kameraabstaende [m]:")
for i in range(5):
    for j in range(i + 1, 5):
        print(f"  Cam{i}-Cam{j}: {np.linalg.norm(cams[i]['C'] - cams[j]['C']):.2f}")
        
# %% Cell 2 - video properties vs. declared image size

import cv2

TRIAL = "squats1"
paths = {f"Cam{i}": VID / f"Cam{i}" / TRIAL / f"{TRIAL}_syncdWithMocap.avi"
         for i in range(5)}

for name, p in paths.items():
    if not p.exists():
        print(f"{name}: MISSING {p.name}")
        continue
    cap = cv2.VideoCapture(str(p))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"{name}: {w}x{h}  {n} frames  {fps:.1f} fps  -> {n/fps:.2f} s")

# reference duration from the OpenSim result of the same trial
ref, meta = read_mot(Path("data/LabValidation_withVideos/subject2/OpenSimData")
                     / "Mocap" / "IK" / f"{TRIAL}.mot")
print(f"\nmocap reference: {len(ref)} rows, {meta['duration_s']:.2f} s, "
      f"{meta['fs_hz']:.1f} Hz")

# %% Cell 3 - one frame through YOLO, visually checked

from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(str(paths["Cam0"]))
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
ok, frame = cap.read()
cap.release()
print("frame read:", ok, frame.shape if ok else None)

res = model(frame, verbose=False)[0]
kp = res.keypoints.xy.cpu().numpy()          # (n_persons, 17, 2)
conf = res.keypoints.conf.cpu().numpy()      # (n_persons, 17)
print("persons detected:", kp.shape[0])

ARM = {5: "L shoulder", 6: "R shoulder", 7: "L elbow", 8: "R elbow",
       9: "L wrist", 10: "R wrist"}
for i, name in ARM.items():
    print(f"  {name:12} {np.round(kp[0, i], 1)}  conf {conf[0, i]:.2f}")

plt.figure(figsize=(4, 7))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.plot(kp[0, :, 0], kp[0, :, 1], "o", ms=4)
for i in ARM:
    plt.plot(kp[0, i, 0], kp[0, i, 1], "s", ms=8)
plt.title("Cam0, frame 200"); plt.axis("off"); plt.tight_layout(); plt.show()

# %% Cell 4 - which cameras see the right arm?

IDX = {"shoulder": 6, "elbow": 8, "wrist": 10}     # right side

frames = {}
for name, p in paths.items():
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
    ok, f = cap.read()
    cap.release()
    frames[name] = f if ok else None

print(f"{'cam':6} {'persons':>8} {'shoulder':>9} {'elbow':>8} {'wrist':>8}")
for name, f in frames.items():
    r = model(f, verbose=False)[0]
    n = r.keypoints.xy.shape[0]
    if n == 0:
        print(f"{name:6} {n:>8}   no detection")
        continue
    c = r.keypoints.conf.cpu().numpy()[0]
    print(f"{name:6} {n:>8} {c[6]:9.2f} {c[8]:8.2f} {c[10]:8.2f}")
    
# %% Cell 5 - keypoints over all frames, two cameras

PAIR = ["Cam0", "Cam2"]
IDX = [6, 8, 10]                      # right shoulder, elbow, wrist


def extract_keypoints(path, indices, model, max_frames=None):
    """Per-frame 2D keypoints from one video.

    Returns (n_frames, len(indices), 2) in pixels and the matching
    confidences. Frames with no detection are filled with NaN so that
    array indices stay aligned with frame numbers.
    # CAP_PROP_FRAME_COUNT comes from the container header and can overstate
    # the number of readable frames. Trailing NaNs are a container artefact,
    # not a detection failure.
    """
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        n = min(n, max_frames)
    xy = np.full((n, len(indices), 2), np.nan)
    cf = np.full((n, len(indices)), np.nan)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, verbose=False)[0]
        if r.keypoints.xy.shape[0] == 0:
            continue
        xy[i] = r.keypoints.xy.cpu().numpy()[0][indices]
        cf[i] = r.keypoints.conf.cpu().numpy()[0][indices]
    cap.release()
    return xy, cf


kps, cfs = {}, {}
for name in PAIR:
    kps[name], cfs[name] = extract_keypoints(paths[name], IDX, model)
    miss = np.isnan(kps[name][:, 0, 0]).sum()
    print(f"{name}: {kps[name].shape[0]} frames, {miss} without detection, "
          f"mean conf {np.nanmean(cfs[name]):.3f}, "
          f"min conf {np.nanmin(cfs[name]):.2f}")
    
# %% Cell 6 - undistort, triangulate, joint angle

from geometry import segment_lengths

cam = {c["name"]: c for c in cams}
P = {n: projection_matrix(cam[n], cam[n]["K"]) for n in PAIR}

valid = ~np.isnan(kps[PAIR[0]][:, 0, 0]) & ~np.isnan(kps[PAIR[1]][:, 0, 0])
print("usable frames:", valid.sum(), "of", len(valid))

uv = {n: undistort(kps[n][valid].reshape(-1, 2), cam[n]["K"], cam[n]["dist"])
      for n in PAIR}
X = triangulate(uv[PAIR[0]], uv[PAIR[1]],
                P[PAIR[0]], P[PAIR[1]]).reshape(-1, 3, 3)

L = segment_lengths(X)
print(f"upper arm [m]: mean {L[:,0].mean():.3f}  sd {L[:,0].std():.3f}  "
      f"range {L[:,0].min():.3f}-{L[:,0].max():.3f}")
print(f"forearm   [m]: mean {L[:,1].mean():.3f}  sd {L[:,1].std():.3f}  "
      f"range {L[:,1].min():.3f}-{L[:,1].max():.3f}")

theta = joint_angle(X[:, 0], X[:, 1], X[:, 2])
print(f"elbow angle (3-point): {theta.min():.1f} .. {theta.max():.1f} deg")

print(f"\ncentroid [m]: {np.round(X.reshape(-1, 3).mean(axis=0), 2)}")

# %% Cell 7 - compare against the OpenCap result for the same trial

t_video = np.arange(len(valid))[valid] / 60.0
flexion_own = 180.0 - theta                     # 3-point angle -> OpenSim convention

BASE_V = Path("data/LabValidation_withVideos/subject2/OpenSimData")
ref, _ = read_mot(BASE_V / "Mocap" / "IK" / f"{TRIAL}.mot")
oc, _ = read_mot(BASE_V / "Video" / "HRNet" / "2-cameras" / "IK" / f"{TRIAL}.mot")

ref_i = np.interp(t_video, ref["time"], ref["elbow_flex_r"])
oc_i = np.interp(t_video, oc["time"], oc["elbow_flex_r"])

for label, series in [("own pipeline", flexion_own), ("OpenCap (HRNet, 2 cams)", oc_i)]:
    e = series - ref_i
    print(f"{label:26} bias {e.mean():+7.2f}  scatter {e.std():6.2f}  "
          f"RMSE {np.sqrt((e**2).mean()):6.2f} deg")

plt.figure(figsize=(9, 4))
plt.plot(t_video, ref_i, label="marker-based reference", lw=1.5)
plt.plot(t_video, oc_i, label="OpenCap (HRNet, 2 cams)", lw=1.2)
plt.plot(t_video, flexion_own, label="own pipeline (YOLO, raw keypoints)", lw=1.2)
plt.xlabel("time [s]"); plt.ylabel("right elbow flexion [deg]")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# %% Cell 8 - extend to several subjects

def analyse_trial(subject, trial, pair=("Cam0", "Cam2"), model=model):
    """Own pipeline vs. OpenCap vs. marker-based reference for one trial.

    Returns a dict of error metrics, or None if any required file is absent.
    Keypoint indices are the right shoulder, elbow and wrist (COCO 6, 8, 10).
    """
    root = Path("data/LabValidation_withVideos") / subject
    vid = root / "VideoData" / "Session0"
    osim = root / "OpenSimData"

    vpaths = {c: vid / c / trial / f"{trial}_syncdWithMocap.avi" for c in pair}
    f_ref = osim / "Mocap" / "IK" / f"{trial}.mot"
    f_oc = osim / "Video" / "HRNet" / "2-cameras" / "IK" / f"{trial}.mot"
    if not all(p.exists() for p in list(vpaths.values()) + [f_ref, f_oc]):
        return None

    cams_s = {c: load_camera(vid / c) for c in pair}
    P_s = {c: projection_matrix(cams_s[c], cams_s[c]["K"]) for c in pair}

    kp_s = {c: extract_keypoints(vpaths[c], [6, 8, 10], model)[0] for c in pair}
    n = min(kp_s[pair[0]].shape[0], kp_s[pair[1]].shape[0])
    ok = (~np.isnan(kp_s[pair[0]][:n, 0, 0])) & (~np.isnan(kp_s[pair[1]][:n, 0, 0]))
    if ok.sum() < 100:
        return None

    uv_s = {c: undistort(kp_s[c][:n][ok].reshape(-1, 2),
                         cams_s[c]["K"], cams_s[c]["dist"]) for c in pair}
    X_s = triangulate(uv_s[pair[0]], uv_s[pair[1]],
                      P_s[pair[0]], P_s[pair[1]]).reshape(-1, 3, 3)

    t = np.arange(n)[ok] / 60.0
    own = 180.0 - joint_angle(X_s[:, 0], X_s[:, 1], X_s[:, 2])

    ref_s, _ = read_mot(f_ref)
    oc_s, _ = read_mot(f_oc)
    ref_i = np.interp(t, ref_s["time"], ref_s["elbow_flex_r"])
    oc_i = np.interp(t, oc_s["time"], oc_s["elbow_flex_r"])

    L_s = segment_lengths(X_s)
    e_own, e_oc = own - ref_i, oc_i - ref_i
    return {"subject": subject, "trial": trial, "frames": int(ok.sum()),
            "own_bias": e_own.mean(), "own_scatter": e_own.std(),
            "oc_bias": e_oc.mean(), "oc_scatter": e_oc.std(),
            "seg_sd_upper_mm": L_s[:, 0].std() * 1000,
            "seg_sd_fore_mm": L_s[:, 1].std() * 1000}


rows = []
for s in ["subject2", "subject3", "subject4", "subject5"]:
    for tr in ["squats1", "STS1"]:
        r = analyse_trial(s, tr)
        if r:
            rows.append(r)
            print(f"{s} {tr:8} n={r['frames']:4}  "
                  f"own {r['own_bias']:+7.2f}/{r['own_scatter']:5.2f}  "
                  f"oc {r['oc_bias']:+7.2f}/{r['oc_scatter']:5.2f}  "
                  f"segSD {r['seg_sd_upper_mm']:4.0f}/{r['seg_sd_fore_mm']:4.0f} mm")

res3 = pd.DataFrame(rows)
print("\n" + res3[["own_bias", "own_scatter", "oc_bias", "oc_scatter",
                   "seg_sd_upper_mm", "seg_sd_fore_mm"]]
      .agg(["mean", "std"]).round(2).to_string())
# -*- coding: utf-8 -*-
"""
02_opencap_exploration.py

Do the synthetic error findings hold on real recordings?

Data: OpenCap laboratory validation set (Uhlrich et al. 2023), downloaded
from SimTK. Ten subjects, five calibrated iPhone cameras, marker-based
reference kinematics, plus markerless results already computed for three
keypoint detectors and 2/3/5 cameras.

The data folder is excluded from version control: the set contains
identifiable video and is subject to a registration agreement.
"""

# %% Cell 0 - environment and paths

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from io_opensim import read_mot, read_pair

ROOT = Path("data/LabValidation_withoutVideos")
SUBJECTS = [f"subject{i}" for i in range(2, 12)]
GAIT = ["walking1.mot", "walking2.mot", "walking3.mot"]
ALL_TASKS = GAIT + ["DJ1.mot", "DJ2.mot", "DJ3.mot",
                    "squats1.mot", "STS1.mot"]

print("subjects found:", sum((ROOT / s).exists() for s in SUBJECTS))


# %% Cell 1 - inspect one file before trusting any number

df, meta = read_mot(ROOT / "subject2" / "OpenSimData" / "Mocap" / "IK" / "walking1.mot")

for k in ["file", "in_degrees", "n_rows_read", "n_cols_read",
          "duration_s", "fs_hz", "dt_jitter_s"]:
    print(f"{k:14}: {meta[k]}")
print("declared      : nRows=%s nColumns=%s" % (meta["nRows"], meta["nColumns"]))
print("arm columns   :", [c for c in df.columns
                          if any(s in c for s in ("arm", "elbow", "pro_sup"))])

# inDegrees=yes, 100 Hz, jitter at rounding level, declared and measured
# counts agree. Note that the model includes arms, so upper-limb
# kinematics are available - not only the lower limb the set is known for.
#
# CONVENTION: OpenSim reports elbow_flex and knee_angle as flexion from
# zero, whereas synthetic.make_arm uses 180 deg for full extension. Bias
# signs are therefore mirrored between the two studies.


# %% Cell 2 - one trial: reference against markerless

BASE = ROOT / "subject2" / "OpenSimData"
ref, vid = read_pair(BASE, "walking1.mot")

for col in ["knee_angle_r", "elbow_flex_r"]:
    truth = ref[col].to_numpy()
    err = vid[col].to_numpy() - truth
    rmse, bias = np.sqrt((err ** 2).mean()), err.mean()
    print(f"\n{col}")
    print(f"  range   : {truth.min():6.1f} .. {truth.max():6.1f} deg")
    print(f"  RMSE    : {rmse:6.2f} deg")
    print(f"  bias    : {bias:+6.2f} deg")
    print(f"  scatter : {np.sqrt(max(rmse**2 - bias**2, 0)):6.2f} deg")

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for ax, col in zip(axes, ["knee_angle_r", "elbow_flex_r"]):
    ax.plot(ref["time"], ref[col], label="marker-based reference", lw=1.5)
    ax.plot(vid["time"], vid[col], label="markerless, HRNet, 2 cameras", lw=1.5)
    ax.set_ylabel(col + " [deg]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
axes[-1].set_xlabel("time [s]")
plt.tight_layout(); plt.show()

# Splitting RMSE into bias and scatter is what makes the elbow result
# readable: almost all of its error is a constant offset, not noise.


# %% Cell 3 - cohort level: does camera count help?

def collect(subjects, trials, joints=("knee_angle_r", "elbow_flex_r"),
            detector="HRNet", cams_list=("2-cameras", "3-cameras", "5-cameras")):
    """Error of the markerless result against the marker-based reference.

    Trials whose time axes do not match are skipped by read_pair rather
    than interpolated, so every row compares identical instants.
    """
    rows = []
    for s in subjects:
        base = ROOT / s / "OpenSimData"
        for t in trials:
            for cams in cams_list:
                pair = read_pair(base, t, detector=detector, cams=cams)
                if pair is None:
                    continue
                ref, vid = pair
                for j in joints:
                    e = vid[j].to_numpy() - ref[j].to_numpy()
                    rows.append({"subject": s, "trial": t, "cams": cams,
                                 "task": "".join(c for c in t.split(".")[0]
                                                 if not c.isdigit()),
                                 "joint": j,
                                 "rom_ref": np.ptp(ref[j].to_numpy()),
                                 "rmse": np.sqrt((e ** 2).mean()),
                                 "bias": e.mean(), "scatter": e.std()})
    return pd.DataFrame(rows)


res_gait = collect(SUBJECTS, GAIT)
print(res_gait.groupby(["joint", "cams"])[["rmse", "bias", "scatter"]]
      .agg(["mean", "std"]).round(2).to_string())
print("\ntrials:", res_gait.trial.count() // 6, " subjects:", res_gait.subject.nunique())

# Knee scatter falls with more cameras; elbow bias and scatter do not
# change at all. A reconstruction-geometry problem improves with
# redundancy, a joint-definition problem does not.


# %% Cell 4 - is the elbow offset task dependent?

res_all = collect(SUBJECTS, ALL_TASKS, cams_list=("2-cameras",))
print(res_all[res_all.joint == "elbow_flex_r"]
      .groupby("task")[["rom_ref", "bias", "scatter"]]
      .agg(["mean", "std", "count"]).round(2).to_string())

# Gait separates clearly from all other tasks (-14.7 deg vs -4.8 to -7.5),
# and the separation exceeds within-task variability. But bias does not
# scale monotonically with range of motion - sit-to-stand has the smallest
# ROM of the three non-gait tasks yet not the largest bias. So amplitude
# alone does not explain it; the error is task specific, and what makes
# gait different cannot be determined from these data.
#
# Consequence for a quality assurance framework: error characteristics
# must be established per movement class. A single validation figure per
# joint is not sufficient, and lower-limb figures do not transfer to the
# upper limb.

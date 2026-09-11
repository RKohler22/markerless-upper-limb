# %% Cell 0 - environment

import sys, os
from pathlib import Path

PROJECT = Path(r"C:\Users\bestizer\Desktop\llui_project")
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
    
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from geometry import projection_matrix, undistort, triangulate, joint_angle
from io_opensim import read_mot

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
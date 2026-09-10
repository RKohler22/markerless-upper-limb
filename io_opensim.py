# -*- coding: utf-8 -*-
"""
io_opensim.py - reading OpenSim .mot / .sto files.

The header of these files declares row count, column count and whether
rotations are in degrees. The reader returns both the declared values and
the values measured from the data, so that any disagreement is visible
rather than assumed away.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def read_mot(path):
    """Read an OpenSim .mot/.sto file with its text header.

    Returns
    -------
    df   : DataFrame, first column 'time' in seconds
    meta : dict with the declared header entries plus measured values
           (n_rows_read, n_cols_read, duration_s, fs_hz, dt_jitter_s)
    """
    path = Path(path)
    meta, n_skip = {}, None
    with open(path) as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s.lower() == "endheader":
                n_skip = i + 1
                break
            if "=" in s:
                k, v = s.split("=", 1)
                meta[k.strip()] = v.strip()
    if n_skip is None:
        raise ValueError(f"no 'endheader' found in {path.name}")

    df = pd.read_csv(path, sep=r"\s+", skiprows=n_skip)

    t = df["time"].to_numpy()
    dt = np.diff(t)
    meta.update({
        "file": path.name,
        "in_degrees": meta.get("inDegrees", "?") == "yes",
        "n_rows_read": len(df),
        "n_cols_read": df.shape[1],
        "duration_s": float(t[-1] - t[0]),
        "fs_hz": float(1.0 / np.median(dt)),
        "dt_jitter_s": float(np.ptp(dt)),
    })
    return df, meta


def read_pair(base, trial, detector="HRNet", cams="2-cameras"):
    """Read the marker-based reference and one markerless result together.

    Returns (ref, vid) or None if either file is missing or the two do not
    share a common time base. Comparing frames across differing time axes
    would measure temporal offset, not reconstruction error - so this
    check is refused rather than silently interpolated.
    """
    f_ref = Path(base) / "Mocap" / "IK" / trial
    f_vid = Path(base) / "Video" / detector / cams / "IK" / trial
    if not (f_ref.exists() and f_vid.exists()):
        return None
    ref, _ = read_mot(f_ref)
    vid, _ = read_mot(f_vid)
    if len(ref) != len(vid) or not np.allclose(ref["time"], vid["time"]):
        return None
    return ref, vid

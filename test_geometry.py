# -*- coding: utf-8 -*-
"""
test_geometry.py - verification of the building blocks in geometry.py.

Run:  python test_geometry.py
All tests must report PASS before working with real data.
"""

import numpy as np

from geometry import (make_board_points, look_at_extrinsics, make_stereo_rig,
                      projection_matrix, project_points, undistort,
                      triangulate, joint_angle, segment_lengths)
from synthetic import make_arm, make_ground_truth_camera

K, DIST, SIZE = make_ground_truth_camera()


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def test_board():
    b = make_board_points()
    return report("board points", b.shape == (54, 3) and abs(b[:, 2]).max() == 0,
                  f"shape={b.shape}")


def test_extrinsics_roundtrip():
    """C = -R.T @ t must invert the extrinsics definition exactly.

    A transpose or sign error here would not raise - it would produce
    plausible looking but systematically wrong 3D points downstream.
    """
    R, t = look_at_extrinsics((1.0, -2.0, 1.2), (0.0, 0.0, 1.2))
    err = np.abs(-R.T @ t - np.array([1.0, -2.0, 1.2])).max()
    return report("extrinsics C = -R.T @ t", err < 1e-12, f"err={err:.1e}")


def test_arm_selfconsistency():
    """The arm model must reproduce its own defining angle, and segment
    lengths must be strictly constant across frames."""
    arm, theta = make_arm()
    err = np.abs(joint_angle(arm[:, 0], arm[:, 1], arm[:, 2]) - theta).max()
    const = np.ptp(segment_lengths(arm), axis=0).max()
    return report("arm: angle and segment lengths",
                  err < 1e-9 and const < 1e-12,
                  f"angle={err:.1e} length={const:.1e}")


def test_triangulation_exact():
    """Without noise the full chain must recover ground truth to machine
    precision. On real data no such reference exists, which is why this
    check has to happen here."""
    arm, theta = make_arm()
    target = arm.reshape(-1, 3).mean(axis=0)
    cams = make_stereo_rig(target)
    P = [projection_matrix(c, K) for c in cams]
    uv = [undistort(project_points(arm.reshape(-1, 3), c, K, DIST), K, DIST)
          for c in cams]
    X = triangulate(uv[0], uv[1], P[0], P[1]).reshape(len(arm), 3, 3)
    perr = np.abs(X - arm).max()
    aerr = np.abs(joint_angle(X[:, 0], X[:, 1], X[:, 2]) - theta).max()
    return report("triangulation without noise", perr < 1e-9 and aerr < 1e-7,
                  f"point={perr:.1e}m angle={aerr:.1e}deg")


def test_noise_linearity():
    """Angular error must scale linearly with keypoint noise (finding 2)."""
    arm, theta = make_arm()
    target = arm.reshape(-1, 3).mean(axis=0)
    cams = make_stereo_rig(target)
    P = [projection_matrix(c, K) for c in cams]
    clean = [project_points(arm.reshape(-1, 3), c, K, DIST) for c in cams]

    rmses = []
    for sigma in (1.0, 3.0):
        errs = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            uv = [undistort(c + rng.normal(0, sigma, c.shape), K, DIST)
                  for c in clean]
            X = triangulate(uv[0], uv[1], P[0], P[1]).reshape(len(arm), 3, 3)
            errs.append(joint_angle(X[:, 0], X[:, 1], X[:, 2]) - theta)
        rmses.append(np.sqrt((np.array(errs) ** 2).mean()))

    ratio = rmses[1] / rmses[0]
    return report("error scales linearly with sigma", 2.7 < ratio < 3.3,
                  f"ratio={ratio:.2f} (expected 3.0)")


if __name__ == "__main__":
    results = [test_board(), test_extrinsics_roundtrip(),
               test_arm_selfconsistency(), test_triangulation_exact(),
               test_noise_linearity()]
    print(f"\n{sum(results)}/{len(results)} tests passed")

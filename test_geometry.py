# -*- coding: utf-8 -*-
"""
test_geometry.py - Verifikation der Bausteine aus geometry.py.

Ausfuehren:  python test_geometry.py
Alle Tests muessen PASS melden, bevor mit echten Daten gearbeitet wird.
"""

import numpy as np
from geometry import (make_board_points, look_at_extrinsics, make_stereo_rig,
                      projection_matrix, project_points, undistort,
                      triangulate, joint_angle, segment_lengths)


K = np.array([[1400.0, 0.0, 960.0],
              [0.0, 1400.0, 540.0],
              [0.0, 0.0, 1.0]])
DIST = np.array([-0.25, 0.08, 0.0, 0.0, 0.0])
SIZE = (1920, 1080)


def make_arm(n_frames=120, l_upper=0.30, l_fore=0.25,
             shoulder=(0.0, 0.0, 1.40), angle_deg=(170.0, 40.0),
             plane_azimuth_deg=35.0):
    """Synthetischer Arm mit bekanntem Ellbogenwinkel."""
    theta = np.radians(np.linspace(*angle_deg, n_frames))
    psi = np.radians(plane_azimuth_deg)
    S = np.tile(np.asarray(shoulder, float), (n_frames, 1))
    E = S + np.array([0.0, 0.0, -l_upper])
    W = E + l_fore * np.stack([np.sin(theta) * np.cos(psi),
                               np.sin(theta) * np.sin(psi),
                               np.cos(theta)], axis=1)
    return np.stack([S, E, W], axis=1), np.degrees(theta)


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def test_board():
    b = make_board_points()
    return report("Board-Punkte", b.shape == (54, 3) and abs(b[:, 2]).max() == 0,
                  f"shape={b.shape}")


def test_extrinsics_roundtrip():
    R, t = look_at_extrinsics((1.0, -2.0, 1.2), (0.0, 0.0, 1.2))
    err = np.abs(-R.T @ t - np.array([1.0, -2.0, 1.2])).max()
    return report("Extrinsics C = -R.T @ t", err < 1e-12, f"err={err:.1e}")


def test_arm_selfconsistency():
    arm, theta = make_arm()
    err = np.abs(joint_angle(arm[:, 0], arm[:, 1], arm[:, 2]) - theta).max()
    L = segment_lengths(arm)
    const = np.ptp(L, axis=0).max()
    return report("Arm: Winkel und Segmentlaengen", err < 1e-9 and const < 1e-12,
                  f"winkel={err:.1e} laenge={const:.1e}")


def test_triangulation_exact():
    """Ohne Rauschen muss die Rekonstruktion die Wahrheit exakt treffen."""
    arm, theta = make_arm()
    target = arm.reshape(-1, 3).mean(axis=0)
    cams = make_stereo_rig(target)
    P = [projection_matrix(c, K) for c in cams]
    uv = [undistort(project_points(arm.reshape(-1, 3), c, K, DIST), K, DIST)
          for c in cams]
    X = triangulate(uv[0], uv[1], P[0], P[1]).reshape(len(arm), 3, 3)
    perr = np.abs(X - arm).max()
    aerr = np.abs(joint_angle(X[:, 0], X[:, 1], X[:, 2]) - theta).max()
    return report("Triangulation ohne Rauschen", perr < 1e-9 and aerr < 1e-7,
                  f"punkt={perr:.1e}m winkel={aerr:.1e}deg")


def test_noise_linearity():
    """Winkelfehler muss linear mit dem Keypoint-Rauschen skalieren."""
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
    return report("Fehler skaliert linear mit sigma", 2.7 < ratio < 3.3,
                  f"ratio={ratio:.2f} (erwartet 3.0)")


if __name__ == "__main__":
    results = [test_board(), test_extrinsics_roundtrip(),
               test_arm_selfconsistency(), test_triangulation_exact(),
               test_noise_linearity()]
    print(f"\n{sum(results)}/{len(results)} Tests bestanden")

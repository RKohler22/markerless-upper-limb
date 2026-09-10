# -*- coding: utf-8 -*-
"""
geometry.py - verified building blocks for the markerless pipeline.

Every function here was checked against known ground truth in study 01.
Do not change anything in this file without re-running test_geometry.py.

Conventions (applied throughout, no exceptions):
    lengths           metres
    angles            degrees at the interface, radians only internally
    image coordinates pixels, u to the right, v downwards
    camera frame      OpenCV: x right, y down, z along the viewing direction
    extrinsics        x_cam = R @ x_world + t,  camera centre C = -R.T @ t
"""

import numpy as np
import cv2


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def make_board_points(cols=9, rows=6, square_m=0.030):
    """3D coordinates of the inner checkerboard corners in board frame (Z = 0)."""
    pts = np.zeros((rows * cols, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return pts * square_m


def calibrate(board, views_noisy, size):
    """Recover intrinsics from 2D-3D correspondences.

    Returns (rms, K, dist). NOTE: rms measures the internal consistency of
    the fit, NOT the validity of the parameters. See README, finding 1.
    """
    objp = [board.astype(np.float32) for _ in views_noisy]
    imgp = [uv.reshape(-1, 1, 2).astype(np.float32) for uv in views_noisy]
    rms, K, dist, _, _ = cv2.calibrateCamera(objp, imgp, size, None, None)
    return rms, K, dist.ravel()


# --------------------------------------------------------------------------
# Camera geometry
# --------------------------------------------------------------------------

def look_at_extrinsics(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """Extrinsics of a camera at cam_pos looking towards target."""
    C = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    return R, -R @ C


def make_stereo_rig(target, radius=1.8, angle_deg=60.0):
    """Two cameras, symmetric about the frontal direction, equal height."""
    target = np.asarray(target, float)
    cams = []
    for sign in (-1, +1):
        az = np.radians(sign * angle_deg / 2)
        C = target + radius * np.array([np.sin(az), -np.cos(az), 0.0])
        R, t = look_at_extrinsics(C, target)
        cams.append({"C": C, "R": R, "t": t})
    return cams


def projection_matrix(cam, K):
    """P = K [R | t]. Valid for UNDISTORTED image points only."""
    return K @ np.hstack([cam["R"], cam["t"].reshape(3, 1)])


def project_points(pts_world, cam, K, dist):
    """(N,3) world points -> (N,2) image points in pixels, distortion included."""
    rvec, _ = cv2.Rodrigues(cam["R"])
    uv, _ = cv2.projectPoints(np.asarray(pts_world, float).reshape(-1, 1, 3),
                              rvec, cam["t"].reshape(3, 1), K, dist)
    return uv.reshape(-1, 2)


def undistort(uv, K, dist):
    """Remove lens distortion, returning pixel coordinates again (P=K)."""
    src = np.asarray(uv, float).reshape(-1, 1, 2)
    return cv2.undistortPoints(src, K, dist, P=K).reshape(-1, 2)


def triangulate(uv0, uv1, P0, P1):
    """DLT triangulation from two views. uv0/uv1 MUST be undistorted."""
    X = cv2.triangulatePoints(P0, P1, uv0.T.astype(float), uv1.T.astype(float))
    return (X[:3] / X[3]).T


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------

def joint_angle(p_prox, p_joint, p_dist):
    """Angle at the middle point of a three-point chain, in degrees.

    Precision degrades near 180 deg (collinearity) and the estimate is
    additionally biased downwards there. See README, finding 3.
    """
    v1 = np.asarray(p_prox) - np.asarray(p_joint)
    v2 = np.asarray(p_dist) - np.asarray(p_joint)
    cos = (np.sum(v1 * v2, axis=-1) /
           (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def segment_lengths(X):
    """Rigid-body check: (n,3,3) -> (n,2) segment lengths in metres.

    On real data these must be approximately constant over time. Their
    dispersion is a quality measure that needs no reference system.
    """
    X = np.asarray(X, float)
    return np.stack([np.linalg.norm(X[:, 0] - X[:, 1], axis=-1),
                     np.linalg.norm(X[:, 2] - X[:, 1], axis=-1)], axis=1)

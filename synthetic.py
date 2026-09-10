# -*- coding: utf-8 -*-
"""
synthetic.py - generators for synthetic scenes with known ground truth.

Kept separate from geometry.py on purpose: geometry.py holds the methods
under test, this file holds the test material. Mixing the two would mean
verifying a pipeline with quantities produced by the same pipeline.
"""

import numpy as np
import cv2

from geometry import make_board_points, project_points  # noqa: F401


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------

def make_ground_truth_camera():
    """Known intrinsics - the 'truth' we later check estimates against.

    Returns
    -------
    K    : (3,3)  camera matrix [fx 0 cx; 0 fy cy; 0 0 1], pixels
    dist : (5,)   distortion [k1, k2, p1, p2, k3]
    size : (w, h) image size in pixels
    """
    K = np.array([[1400.0,    0.0, 960.0],
                  [   0.0, 1400.0, 540.0],
                  [   0.0,    0.0,   1.0]])
    dist = np.array([-0.25, 0.08, 0.0, 0.0, 0.0])
    size = (1920, 1080)
    return K, dist, size


def project_pose(obj_pts, K, dist, rvec, tvec):
    """Project 3D points into one camera image.

    rvec : Rodrigues vector (direction = rotation axis, length = angle in rad)
    tvec : translation board -> camera, in metres

    Returns (N, 2) image coordinates in pixels.
    """
    img_pts, _ = cv2.projectPoints(obj_pts,
                                   np.asarray(rvec, float),
                                   np.asarray(tvec, float),
                                   K, dist)
    return img_pts.reshape(-1, 2)


# --------------------------------------------------------------------------
# Checkerboard pose sets
# --------------------------------------------------------------------------

def board_in_front(obj_pts, rvec, tvec, z_min=0.15):
    """Are all board points in front of the camera? Otherwise projectPoints
    returns meaningless values without raising."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    pts_cam = (R @ obj_pts.T).T + np.asarray(tvec, float)
    return pts_cam[:, 2].min() > z_min


def pose_in_image(uv, size, margin=20):
    """Are all corners inside the image, with a margin?"""
    w, h = size
    return (uv[:, 0].min() > margin and uv[:, 0].max() < w - margin and
            uv[:, 1].min() > margin and uv[:, 1].max() < h - margin)


def make_pose_set(board, K, dist, size, n=20, seed=0, tilt_max=0.6):
    """Rejection sampling: random board poses, only valid ones are kept.

    tilt_max controls the range of the Rodrigues components and therefore
    how strongly the board is inclined. Low values make the focal length
    poorly observable - see README, finding 1.
    """
    rng = np.random.default_rng(seed)
    poses, views, tries = [], [], 0
    while len(poses) < n and tries < 5000:
        tries += 1
        rvec = rng.uniform(-tilt_max, tilt_max, 3)
        tvec = np.array([-0.12 + rng.uniform(-0.10, 0.10),
                         -0.075 + rng.uniform(-0.07, 0.07),
                         rng.uniform(0.50, 0.90)])
        if not board_in_front(board, rvec, tvec):
            continue
        uv = project_pose(board, K, dist, rvec, tvec)
        if not pose_in_image(uv, size):
            continue
        poses.append((rvec, tvec))
        views.append(uv)
    return poses, views, tries


def add_detection_noise(views, sigma_px=0.3, seed=1):
    """Subpixel corner detection is not exact. Realistic range: 0.2-0.5 px."""
    rng = np.random.default_rng(seed)
    return [uv + rng.normal(0, sigma_px, uv.shape) for uv in views]


# --------------------------------------------------------------------------
# Limb model
# --------------------------------------------------------------------------

def make_arm(n_frames=120, l_upper=0.30, l_fore=0.25,
             shoulder=(0.0, 0.0, 1.40), angle_deg=(170.0, 40.0),
             plane_azimuth_deg=35.0):
    """Planar elbow flexion, plane orientation set by plane_azimuth_deg.

    The upper arm points straight down, so the elbow angle equals theta
    exactly. Segment lengths are invariant to plane_azimuth_deg because
    cos^2 + sin^2 = 1 - both properties are asserted in test_geometry.py.

    Returns
    -------
    pts   : (n_frames, 3, 3) - frames x [shoulder, elbow, wrist] x xyz [m]
    theta : (n_frames,)      - true elbow angle in degrees (180 = extended)

    NOTE the convention: 180 deg is full extension here, whereas OpenSim
    reports elbow_flex as flexion from zero. Signs of any bias flip
    accordingly when comparing against OpenSim output.
    """
    theta = np.radians(np.linspace(*angle_deg, n_frames))
    psi = np.radians(plane_azimuth_deg)
    S = np.tile(np.asarray(shoulder, float), (n_frames, 1))
    E = S + np.array([0.0, 0.0, -l_upper])
    W = E + l_fore * np.stack([np.sin(theta) * np.cos(psi),
                               np.sin(theta) * np.sin(psi),
                               np.cos(theta)], axis=1)
    return np.stack([S, E, W], axis=1), np.degrees(theta)

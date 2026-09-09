# -*- coding: utf-8 -*-
"""
geometry.py - verifizierte Bausteine fuer die Markerless-Pipeline.

Jede Funktion hier wurde in Block 1 gegen bekannte Ground Truth geprueft.
Nichts in dieser Datei aendern, ohne die Tests in test_geometry.py neu
laufen zu lassen.

Konventionen (durchgehend, keine Ausnahmen):
    Laengen        Meter
    Winkel         Grad nach aussen, Radiant nur intern
    Bildkoordinaten  Pixel, u nach rechts, v nach unten
    Kamera-KS      OpenCV: x rechts, y unten, z in Blickrichtung
    Extrinsics     x_cam = R @ x_world + t,  Kamerazentrum C = -R.T @ t
"""

import numpy as np
import cv2


# --------------------------------------------------------------------------
# Kalibrierung
# --------------------------------------------------------------------------

def make_board_points(cols=9, rows=6, square_m=0.030):
    """3D-Koordinaten der inneren Schachbrettecken im Board-KS (Z = 0)."""
    pts = np.zeros((rows * cols, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return pts * square_m


def calibrate(board, views_noisy, size):
    """Intrinsics aus 2D-3D-Korrespondenzen.

    Returns (rms, K, dist). ACHTUNG: rms ist ein Mass fuer die innere
    Konsistenz der Anpassung, NICHT fuer die Richtigkeit der Parameter.
    Siehe README, Befund 1.
    """
    objp = [board.astype(np.float32) for _ in views_noisy]
    imgp = [uv.reshape(-1, 1, 2).astype(np.float32) for uv in views_noisy]
    rms, K, dist, _, _ = cv2.calibrateCamera(objp, imgp, size, None, None)
    return rms, K, dist.ravel()


# --------------------------------------------------------------------------
# Kamerageometrie
# --------------------------------------------------------------------------

def look_at_extrinsics(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """Extrinsics einer Kamera, die von cam_pos auf target blickt."""
    C = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    return R, -R @ C


def make_stereo_rig(target, radius=1.8, angle_deg=60.0):
    """Zwei Kameras, symmetrisch um die Frontalrichtung, gleiche Hoehe."""
    target = np.asarray(target, float)
    cams = []
    for sign in (-1, +1):
        az = np.radians(sign * angle_deg / 2)
        C = target + radius * np.array([np.sin(az), -np.cos(az), 0.0])
        R, t = look_at_extrinsics(C, target)
        cams.append({"C": C, "R": R, "t": t})
    return cams


def projection_matrix(cam, K):
    """P = K [R | t]. Gilt nur fuer ENTZERRTE Bildpunkte."""
    return K @ np.hstack([cam["R"], cam["t"].reshape(3, 1)])


def project_points(pts_world, cam, K, dist):
    """(N,3) Weltpunkte -> (N,2) Bildpunkte in Pixel, mit Verzeichnung."""
    rvec, _ = cv2.Rodrigues(cam["R"])
    uv, _ = cv2.projectPoints(np.asarray(pts_world, float).reshape(-1, 1, 3),
                              rvec, cam["t"].reshape(3, 1), K, dist)
    return uv.reshape(-1, 2)


def undistort(uv, K, dist):
    """Verzeichnung entfernen, Ergebnis wieder in Pixel (P=K)."""
    src = np.asarray(uv, float).reshape(-1, 1, 2)
    return cv2.undistortPoints(src, K, dist, P=K).reshape(-1, 2)


def triangulate(uv0, uv1, P0, P1):
    """DLT-Triangulation. uv0/uv1 MUESSEN entzerrt sein."""
    X = cv2.triangulatePoints(P0, P1, uv0.T.astype(float), uv1.T.astype(float))
    return (X[:3] / X[3]).T


# --------------------------------------------------------------------------
# Kinematik
# --------------------------------------------------------------------------

def joint_angle(p_prox, p_joint, p_dist):
    """Winkel am mittleren Punkt aus drei 3D-Punkten, in Grad.

    Genauigkeit degradiert nahe 180 Grad (Kollinearitaet), zusaetzlich
    mit systematischem Bias nach unten. Siehe README, Befund 3.
    """
    v1 = np.asarray(p_prox) - np.asarray(p_joint)
    v2 = np.asarray(p_dist) - np.asarray(p_joint)
    cos = (np.sum(v1 * v2, axis=-1) /
           (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def segment_lengths(X):
    """Starrkoerperpruefung: (n,3,3) -> (n,2) Segmentlaengen in Meter.

    Bei echten Daten muessen diese ueber die Zeit annaehernd konstant sein.
    Ihre Streuung ist ein Qualitaetsmass, das ohne Referenzsystem auskommt.
    """
    X = np.asarray(X, float)
    return np.stack([np.linalg.norm(X[:, 0] - X[:, 1], axis=-1),
                     np.linalg.norm(X[:, 2] - X[:, 1], axis=-1)], axis=1)

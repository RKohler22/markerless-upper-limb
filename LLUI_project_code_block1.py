# -*- coding: utf-8 -*-
"""
Block 1 - Kamerakalibrierung (synthetisch)
Projekt: Markerless-Pipeline, Validierung gegen bekannte Ground Truth

Bedienung in Spyder:
    Strg+Enter   Zelle ausfuehren
    Shift+Enter  Zelle ausfuehren und zur naechsten springen
    Strg+.       Kernel neu starten (bei unklarem Zustand IMMER zuerst)
"""

# %% Zelle 0 - Umgebung pruefen

import numpy as np
import cv2
import matplotlib.pyplot as plt

print("OpenCV:", cv2.__version__)
print("NumPy :", np.__version__)


# %% Zelle 1 - Szenendefinition (wiederverwendbare Funktionen)

def make_board_points(cols=9, rows=6, square_m=0.030):
    """3D-Koordinaten der inneren Schachbrettecken im Board-KS (Z = 0).

    Returns
    -------
    (cols*rows, 3) float32 - Einheit: Meter
    """
    pts = np.zeros((rows * cols, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return pts * square_m


def make_ground_truth_camera():
    """Bekannte Intrinsics - die 'Wahrheit', gegen die wir spaeter pruefen.

    Returns
    -------
    K    : (3,3)  Kameramatrix [fx 0 cx; 0 fy cy; 0 0 1], Einheit Pixel
    dist : (5,)   Verzeichnung [k1, k2, p1, p2, k3]
    size : (w, h) Bildgroesse in Pixel
    """
    K = np.array([[1400.0,    0.0, 960.0],
                  [   0.0, 1400.0, 540.0],
                  [   0.0,    0.0,   1.0]])
    dist = np.array([-0.25, 0.08, 0.0, 0.0, 0.0])
    size = (1920, 1080)
    return K, dist, size


def project_pose(obj_pts, K, dist, rvec, tvec):
    """Projiziert 3D-Punkte in das Bild einer Kamera.

    rvec : Rodrigues-Vektor (Richtung = Drehachse, Laenge = Winkel in rad)
    tvec : Translation Board -> Kamera, Einheit Meter

    Returns
    -------
    (N, 2) float - Bildkoordinaten in Pixel
    """
    img_pts, _ = cv2.projectPoints(obj_pts,
                                   np.asarray(rvec, float),
                                   np.asarray(tvec, float),
                                   K, dist)
    return img_pts.reshape(-1, 2)


def plot_projection(uv, size, title="Projektion"):
    """Visuelle Kontrolle: liegen die Ecken im Bild und ist das Raster
    plausibel verzerrt?"""
    w, h = size
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(uv[:, 0], uv[:, 1], "o", ms=3)
    ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False, lw=1.5))
    ax.set_xlim(-100, w + 100)
    ax.set_ylim(h + 100, -100)          # Bildkoordinaten: y zeigt nach unten
    ax.set_aspect("equal")
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# %% Zelle 2 - Verifikation: eine einzelne Pose

board = make_board_points()
K, dist, size = make_ground_truth_camera()

uv = project_pose(board, K, dist,
                  rvec=[0.30, -0.40, 0.05],      # gekippt
                  tvec=[-0.12, -0.075, 0.60])    # 60 cm vor der Kamera

print("Board-Punkte :", board.shape)
print("Bildpunkte   :", uv.shape)
print("min [px]     :", uv.min(0).round(1))
print("max [px]     :", uv.max(0).round(1))
print("Breite  [px] :", round(np.ptp(uv[:, 0]), 1))
print("Hoehe   [px] :", round(np.ptp(uv[:, 1]), 1))

plot_projection(uv, size, "Zelle 2 - eine Pose, gekippt")

# Erwartet:
#   board (54, 3), uv (54, 2)
#   alle Koordinaten innerhalb 0..1920 bzw. 0..1080
#   Brett grob 400-700 px breit
#   im Plot: perspektivisch verzerrtes Raster, sichtbar gekippt

# %% Zelle 3 - 20 gueltige Posen erzeugen

def board_in_front(obj_pts, rvec, tvec, z_min=0.15):
    """Liegen alle Brettpunkte vor der Kamera? Sonst liefert projectPoints Unsinn."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    pts_cam = (R @ obj_pts.T).T + np.asarray(tvec, float)
    return pts_cam[:, 2].min() > z_min


def pose_in_image(uv, size, margin=20):
    """Liegen alle Ecken mit Rand im Bild?"""
    w, h = size
    return (uv[:, 0].min() > margin and uv[:, 0].max() < w - margin and
            uv[:, 1].min() > margin and uv[:, 1].max() < h - margin)


def make_pose_set(board, K, dist, size, n=20, seed=0, tilt_max=0.6):
    """Rejection Sampling: zufaellige Posen, nur gueltige werden behalten."""
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


poses, views, tries = make_pose_set(board, K, dist, size, n=20, seed=0)

tilts = np.degrees([np.linalg.norm(rv) for rv, _ in poses])
alluv = np.vstack(views)

print("Posen        :", len(poses), "aus", tries, "Versuchen")
print("Kippwinkel   : min %.1f  median %.1f  max %.1f Grad"
      % (tilts.min(), np.median(tilts), tilts.max()))
print("Abdeckung u  : %.0f .. %.0f von %d" % (alluv[:, 0].min(), alluv[:, 0].max(), size[0]))
print("Abdeckung v  : %.0f .. %.0f von %d" % (alluv[:, 1].min(), alluv[:, 1].max(), size[1]))

fig, ax = plt.subplots(figsize=(8, 5))
for uv_i in views:
    ax.plot(uv_i[:, 0], uv_i[:, 1], ".", ms=2)
ax.add_patch(plt.Rectangle((0, 0), size[0], size[1], fill=False, lw=1.5))
ax.set_xlim(-50, size[0] + 50); ax.set_ylim(size[1] + 50, -50)
ax.set_aspect("equal"); ax.set_title("Zelle 3 - Abdeckung aller 20 Posen")
plt.tight_layout(); plt.show()

# %% Zelle 4 - Kalibrierung aus verrauschten Detektionen

def add_detection_noise(views, sigma_px=0.3, seed=1):
    """Subpixel-Eckendetektion ist nicht exakt. Realistisch: 0.2-0.5 px."""
    rng = np.random.default_rng(seed)
    return [uv + rng.normal(0, sigma_px, uv.shape) for uv in views]


def calibrate(board, views_noisy, size):
    """Rueckrechnung der Intrinsics aus 2D-3D-Korrespondenzen."""
    objp = [board.astype(np.float32) for _ in views_noisy]
    imgp = [uv.reshape(-1, 1, 2).astype(np.float32) for uv in views_noisy]
    rms, K_est, dist_est, rvecs, tvecs = cv2.calibrateCamera(
        objp, imgp, size, None, None)
    return rms, K_est, dist_est.ravel()


def compare_to_truth(K_est, dist_est, K, dist):
    """Der eigentliche Punkt des synthetischen Ansatzes: wir kennen die Wahrheit."""
    names = ["fx", "fy", "cx", "cy"]
    true = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    est = [K_est[0, 0], K_est[1, 1], K_est[0, 2], K_est[1, 2]]
    print(f"{'Param':6} {'wahr':>10} {'geschaetzt':>12} {'Abw.':>10} {'Abw. %':>8}")
    for n, t, e in zip(names, true, est):
        print(f"{n:6} {t:10.2f} {e:12.2f} {e-t:10.2f} {100*(e-t)/t:8.2f}")
    for i, n in enumerate(["k1", "k2", "p1", "p2", "k3"]):
        print(f"{n:6} {dist[i]:10.4f} {dist_est[i]:12.4f} "
              f"{dist_est[i]-dist[i]:10.4f}")


SIGMA = 0.3
views_noisy = add_detection_noise(views, sigma_px=SIGMA)
rms, K_est, dist_est = calibrate(board, views_noisy, size)

print(f"Injiziertes Rauschen : {SIGMA:.2f} px")
print(f"RMS-Reprojektion     : {rms:.3f} px\n")
compare_to_truth(K_est, dist_est, K, dist)

# %% Zelle 5 - Warum Verkippung noetig ist

def run_experiment(tilt_max, n=20, sigma=0.3, seed=0):
    """Eine Kalibrierung mit vorgegebenem Kippbereich."""
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


print(f"{'tilt_max':>9} {'RMS [px]':>9} {'fx':>9} {'fx Abw.%':>9} {'k1':>9}")
for t in [0.03, 0.10, 0.20, 0.40, 0.60]:
    r = run_experiment(t)
    if r:
        print(f"{r['tilt_max']:9.2f} {r['rms']:9.3f} {r['fx']:9.1f} "
              f"{r['fx_err_pct']:9.2f} {r['k1']:9.4f}")
        
        
# %% Zelle 6 - Streuung ueber mehrere Ziehungen

def sweep(tilt_values, seeds=range(10)):
    """Wiederholt jedes Experiment mit verschiedenen Zufallsziehungen."""
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

print(f"{'tilt_max':>9} {'n':>3} {'|fx-Abw.| %':>12} {'SD':>7} {'max':>7} {'RMS [px]':>9}")
for r in rows:
    print(f"{r['tilt_max']:9.2f} {r['n']:3d} {r['err_mean']:12.2f} "
          f"{r['err_std']:7.2f} {r['err_max']:7.2f} {r['rms_mean']:9.3f}")
    
# %% Zelle 7 - Synthetischer Arm, Ellbogenwinkel als Ground Truth

def make_arm(n_frames=120, l_upper=0.30, l_fore=0.25,
             shoulder=(0.0, 0.0, 1.40), angle_deg=(170.0, 40.0),
             plane_azimuth_deg=35.0):
    """Ebene Ellbogenbeugung in der x-z-Ebene.

    Returns
    -------
    pts   : (n_frames, 3, 3) - Frames x [Schulter, Ellbogen, Handgelenk] x xyz [m]
    theta : (n_frames,)      - wahrer Ellbogenwinkel in Grad (180 = gestreckt)
    """
    theta = np.radians(np.linspace(*angle_deg, n_frames))
    psi = np.radians(plane_azimuth_deg)
    S = np.tile(np.asarray(shoulder, float), (n_frames, 1))
    E = S + np.array([0.0, 0.0, -l_upper])          # Oberarm senkrecht nach unten
    W = E + l_fore * np.stack([np.sin(theta) * np.cos(psi),
                           np.sin(theta) * np.sin(psi),
                           np.cos(theta)], axis=1)
    return np.stack([S, E, W], axis=1), np.degrees(theta)


def joint_angle(p_prox, p_joint, p_dist):
    """Winkel am mittleren Punkt, aus drei 3D-Punkten. Achtet auf Broadcasting."""
    v1 = np.asarray(p_prox) - np.asarray(p_joint)
    v2 = np.asarray(p_dist) - np.asarray(p_joint)
    cos = (np.sum(v1 * v2, axis=-1) /
           (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


arm, theta_true = make_arm()

theta_check = joint_angle(arm[:, 0], arm[:, 1], arm[:, 2])
len_upper = np.linalg.norm(arm[:, 0] - arm[:, 1], axis=-1)
len_fore = np.linalg.norm(arm[:, 2] - arm[:, 1], axis=-1)

print("Form         :", arm.shape)
print("Winkelbereich: %.1f .. %.1f Grad" % (theta_true.min(), theta_true.max()))
print("Rueckrechnung: max. Abweichung %.2e Grad"
      % np.abs(theta_check - theta_true).max())
print("Oberarm  [m] : %.4f .. %.4f" % (len_upper.min(), len_upper.max()))
print("Unterarm [m] : %.4f .. %.4f" % (len_fore.min(), len_fore.max()))

# %% Zelle 8 - Stereo-Rig: Extrinsics und Projektion

def look_at_extrinsics(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """Extrinsics einer Kamera, die von cam_pos auf target blickt.
    OpenCV-Konvention: x rechts, y unten, z in Blickrichtung.
    Returns R (3,3), t (3,) mit  x_cam = R @ x_world + t
    """
    C = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float)); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    return R, -R @ C


def make_stereo_rig(target, radius=2.5, angle_deg=60.0):
    """Zwei Kameras, symmetrisch um die Frontalrichtung, gleiche Hoehe."""
    target = np.asarray(target, float)
    cams = []
    for sign in (-1, +1):
        az = np.radians(sign * angle_deg / 2)
        C = target + radius * np.array([np.sin(az), -np.cos(az), 0.0])
        R, t = look_at_extrinsics(C, target)
        cams.append({"C": C, "R": R, "t": t})
    return cams


def project_points(pts_world, cam, K, dist):
    """Projiziert (N,3) Weltpunkte in eine Kamera. Returns (N,2) in Pixel."""
    rvec, _ = cv2.Rodrigues(cam["R"])
    uv, _ = cv2.projectPoints(np.asarray(pts_world, float).reshape(-1, 1, 3),
                              rvec, cam["t"].reshape(3, 1), K, dist)
    return uv.reshape(-1, 2)


target = arm.reshape(-1, 3).mean(axis=0)
cams = make_stereo_rig(target, radius=1.8, angle_deg=60.0)

for i, cam in enumerate(cams):
    C_rec = -cam["R"].T @ cam["t"]
    print(f"Kamera {i}: C = {np.round(cam['C'], 3)}   "
          f"Rueckrechnung Abw. {np.abs(C_rec - cam['C']).max():.2e}")

views_arm = []
for i, cam in enumerate(cams):
    uv = project_points(arm.reshape(-1, 3), cam, K, dist).reshape(len(arm), 3, 2)
    views_arm.append(uv)
    f = uv.reshape(-1, 2)
    ok = ((f[:, 0] > 0) & (f[:, 0] < size[0]) &
          (f[:, 1] > 0) & (f[:, 1] < size[1])).all()
    print(f"Kamera {i}: u {f[:,0].min():.0f}..{f[:,0].max():.0f}  "
          f"v {f[:,1].min():.0f}..{f[:,1].max():.0f}  alle im Bild: {ok}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, uv, i in zip(axes, views_arm, range(2)):
    for k in range(0, len(arm), 10):                 # jede 10. Pose
        ax.plot(uv[k, :, 0], uv[k, :, 1], "-o", ms=3, lw=1)
    ax.add_patch(plt.Rectangle((0, 0), size[0], size[1], fill=False, lw=1.2))
    ax.set_xlim(0, size[0]); ax.set_ylim(size[1], 0)
    ax.set_aspect("equal"); ax.set_title(f"Kamera {i}")
plt.tight_layout(); plt.show()

for i, uv in enumerate(views_arm):
    w = uv[:, 2, :]
    print(f"Kamera {i}: Handgelenk u-Spanne {np.ptp(w[:,0]):5.0f} px   "
          f"v-Spanne {np.ptp(w[:,1]):5.0f} px")
    
# %% Zelle 9 - DLT-Triangulation und Winkelfehler

def projection_matrix(cam, K):
    """P = K [R | t], gueltig fuer ENTZERRTE, normalisierte Punkte."""
    return K @ np.hstack([cam["R"], cam["t"].reshape(3, 1)])


def undistort(uv, K, dist):
    """Entfernt die Verzeichnung, gibt Pixelkoordinaten zurueck (P=K[R|t])."""
    src = np.asarray(uv, float).reshape(-1, 1, 2)
    return cv2.undistortPoints(src, K, dist, P=K).reshape(-1, 2)


def triangulate(uv0, uv1, P0, P1):
    """DLT-Triangulation zweier Ansichten. uv müssen entzerrt sein."""
    X = cv2.triangulatePoints(P0, P1, uv0.T.astype(float), uv1.T.astype(float))
    return (X[:3] / X[3]).T


def reconstruct_arm(views_arm, cams, K, dist, sigma_px=0.0, seed=0):
    """Kompletter Weg: Rauschen -> Entzerrung -> Triangulation -> Winkel."""
    rng = np.random.default_rng(seed)
    n = views_arm[0].shape[0]
    P = [projection_matrix(c, K) for c in cams]

    uv = [v.reshape(-1, 2) + rng.normal(0, sigma_px, (n * 3, 2))
          for v in views_arm]
    uvu = [undistort(u, K, dist) for u in uv]

    X = triangulate(uvu[0], uvu[1], P[0], P[1]).reshape(n, 3, 3)
    theta = joint_angle(X[:, 0], X[:, 1], X[:, 2])
    return X, theta


# --- Kontrolle ohne Rauschen: muss die Wahrheit exakt treffen
X0, theta0 = reconstruct_arm(views_arm, cams, K, dist, sigma_px=0.0)

print("OHNE RAUSCHEN")
print("  Punktfehler  max : %.2e m" % np.abs(X0 - arm).max())
print("  Winkelfehler max : %.2e Grad" % np.abs(theta0 - theta_true).max())
print("  Oberarm  [m] : %.4f .. %.4f" % (
    np.linalg.norm(X0[:, 0] - X0[:, 1], axis=-1).min(),
    np.linalg.norm(X0[:, 0] - X0[:, 1], axis=-1).max()))

# --- Mit realistischem Keypoint-Rauschen
for s in [1.0, 3.0, 5.0]:
    errs = []
    for seed in range(20):
        _, th = reconstruct_arm(views_arm, cams, K, dist, sigma_px=s, seed=seed)
        errs.append(th - theta_true)
    e = np.array(errs)
    print(f"\nsigma = {s:.1f} px")
    print(f"  Bias      : {e.mean():+.2f} Grad")
    print(f"  RMSE      : {np.sqrt((e**2).mean()):.2f} Grad")
    print(f"  max |Fehler|: {np.abs(e).max():.2f} Grad")
    
# %% Zelle 10 - Fehlerprofil ueber die Bewegung

def error_profile(views_arm, cams, K, dist, sigma_px=3.0, n_seeds=100):
    errs = [reconstruct_arm(views_arm, cams, K, dist, sigma_px=sigma_px,
                            seed=s)[1] - theta_true for s in range(n_seeds)]
    e = np.array(errs)
    return e.mean(0), np.sqrt((e ** 2).mean(0))


bias_f, rmse_f = error_profile(views_arm, cams, K, dist, sigma_px=3.0)

print(f"RMSE ueber Frames: min {rmse_f.min():.2f}  median "
      f"{np.median(rmse_f):.2f}  max {rmse_f.max():.2f} Grad")
print(f"Schlechtester Frame bei theta = {theta_true[rmse_f.argmax()]:.0f} Grad")
print(f"Bester Frame bei        theta = {theta_true[rmse_f.argmin()]:.0f} Grad")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(theta_true, rmse_f, lw=1.5)
ax.set_xlabel("wahrer Ellbogenwinkel [Grad]")
ax.set_ylabel("RMSE [Grad]")
ax.set_title("Fehlerprofil bei sigma = 3 px")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

i_ext = theta_true > 160
i_mid = (theta_true > 70) & (theta_true < 110)
print(f"Bias bei Streckung (>160 Grad): {bias_f[i_ext].mean():+.3f} Grad")
print(f"Bias mittlerer Bereich        : {bias_f[i_mid].mean():+.3f} Grad")
print(f"RMSE Streckung / RMSE Mitte   : "
      f"{rmse_f[i_ext].mean()/rmse_f[i_mid].mean():.2f}")

# %% Zelle 11 - Wie haengt der Fehler von der Bewegungsrichtung ab?

def eval_config(plane_azimuth_deg, radius=1.8, rig_angle=60.0,
                sigma_px=3.0, n_seeds=50):
    """Eine vollstaendige Konfiguration: Arm erzeugen, projizieren,
    rekonstruieren, Fehler gegen die Wahrheit."""
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


azimuths = [0, 15, 30, 45, 60, 75, 90]
res = [eval_config(a) for a in azimuths]

print(f"{'Azimut':>7} {'im Bild':>8} {'RMSE':>8} {'RMSE Ext':>9} {'Bias Ext':>9}")
for r in res:
    print(f"{r['az']:7.0f} {str(r['inside']):>8} {r['rmse']:8.2f} "
          f"{r['rmse_ext']:9.2f} {r['bias_ext']:+9.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([r["az"] for r in res], [r["rmse"] for r in res], "-o", label="gesamt")
ax.plot([r["az"] for r in res], [r["rmse_ext"] for r in res], "-s", label="nahe Streckung")
ax.set_xlabel("Azimut der Bewegungsebene [Grad]")
ax.set_ylabel("RMSE [Grad]")
ax.set_title("Rig-Konditionierung bei sigma = 3 px")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""Objectif 1 — Segmentation des anomalies routières (DeepLabV3+).

Localise au pixel près les objets inattendus sur la route, les numérote, et décrit chaque zone.
Modèle : DeepLabV3+ / ResNet34 pré-entraîné ImageNet, entraîné sur RoadAnomaly, exporté en ONNX
(fichiers model.onnx et config.json dans modeles/objectif_1/).

Cette page a sa propre interface (gabarit templates/objectif_1.html) : elle définit donc
GABARIT et la fonction analyser(), au lieu de la page standard des autres objectifs.
"""
import json
import time
import uuid
from collections import OrderedDict

import cv2
import numpy as np

from .commun import dossier_modele, en_base64, pct

INFO = {
    "bouton": "Segmentation",
    "titre": "Segmentation des anomalies routières",
    "membre": "Mon objectif",
    "description": ("Déposez une photo de route : le modèle repère au pixel près les objets inattendus "
                    "(animaux, rochers, cônes, pneus, obstacles) et les numérote. Cliquez sur une zone "
                    "du tableau pour l'examiner de près."),
}
PRET = True
GABARIT = "objectif_1.html"
FICHIERS_REQUIS = ("model.onnx", "config.json")

DOSSIER = dossier_modele(1)

# Couleurs (RVB) utilisées sur les images
CORAIL = np.array([249, 87, 56])       # remplissage des anomalies
CORAIL_FONCE = (190, 45, 20)           # contours
BLANC = (255, 255, 255)                # cadres et numéros


# ----------------------------------------------------------------------------- #
# Chargement
# ----------------------------------------------------------------------------- #
def charger():
    """Charge le modèle ONNX et sa configuration (une seule fois, au démarrage)."""
    import onnxruntime as ort
    cfg = json.load(open(DOSSIER / "config.json", encoding="utf-8"))
    session = ort.InferenceSession(str(DOSSIER / "model.onnx"), providers=ort.get_available_providers())
    return {"cfg": cfg, "session": session,
            "mean": np.array(cfg["mean"], np.float32), "std": np.array(cfg["std"], np.float32)}


def pastilles(modele):
    cfg = modele["cfg"]
    return [cfg["label"].split(". ", 1)[-1], f"AP {cfg['cv_AP']:.3f}",
            f"IoU {cfg['cv_IoU_mean']:.3f} ± {cfg['cv_IoU_std']:.3f}", "Validation croisée 5 plis"]


def seuil_defaut(modele):
    return round(float(modele["cfg"]["threshold"]), 2)


# ----------------------------------------------------------------------------- #
# Modèle
# ----------------------------------------------------------------------------- #
def predict_proba(img, modele):
    """Image RGB (H, W, 3) uint8 de taille quelconque -> probabilité d'anomalie (H, W)."""
    cfg = modele["cfg"]
    H, W = img.shape[:2]
    w = cfg["work_width"]
    h = max(32, int(round(H * w / W)))
    x = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    x = np.pad(x, ((0, (-h) % 32), (0, (-w) % 32), (0, 0)))
    x = ((x.astype(np.float32) / 255 - modele["mean"]) / modele["std"]).transpose(2, 0, 1)[None]
    logits = modele["session"].run(None, {"image": x})[0][0, 0, :h, :w]
    p = 1 / (1 + np.exp(-logits))
    return cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)


def position(cx, cy, W, H):
    horiz = "à gauche" if cx < W / 3 else ("au centre" if cx < 2 * W / 3 else "à droite")
    vert = "en haut" if cy < H / 3 else ("au milieu" if cy < 2 * H / 3 else "en bas")
    return f"{vert}, {horiz}"


def niveau(c):
    return "élevée" if c >= 0.8 else ("moyenne" if c >= 0.5 else "faible")


def trouver_zones(p, seuil):
    mask = (p >= seuil).astype(np.uint8)
    H, W = mask.shape
    k, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(30, 1e-4 * H * W)
    zones = []
    for i in range(1, k):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a < min_area:
            continue
        probs = p[lab == i]
        zones.append(dict(id=i, x=x, y=y, w=w, h=h, aire=a, part=a / (H * W),
                          conf_moy=float(probs.mean()), conf_max=float(probs.max()),
                          pos=position(cent[i][0], cent[i][1], W, H)))
    zones.sort(key=lambda z: -z["aire"])
    garde = np.isin(lab, [z["id"] for z in zones])       # on retire les miettes trop petites
    return garde.astype(np.uint8), zones


# ----------------------------------------------------------------------------- #
# Dessin
# ----------------------------------------------------------------------------- #
def dessiner(img, mask, zones):
    vis = img.copy()
    zone = mask.astype(bool)
    vis[zone] = (0.45 * vis[zone] + 0.55 * CORAIL).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape
    ep = max(2, W // 450)
    cv2.drawContours(vis, contours, -1, CORAIL_FONCE, ep)
    taille = max(0.5, W / 1400)
    for i, z in enumerate(zones):
        cv2.rectangle(vis, (z["x"], z["y"]), (z["x"] + z["w"], z["y"] + z["h"]), BLANC, ep)
        txt = str(i + 1)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, taille, max(1, ep))
        y0 = max(th + 8, z["y"])
        cv2.rectangle(vis, (z["x"], y0 - th - 8), (z["x"] + tw + 10, y0), BLANC, -1)
        cv2.putText(vis, txt, (z["x"] + 5, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, taille, (30, 41, 59), max(1, ep),
                    cv2.LINE_AA)
    return vis


def carte_chaleur(p):
    return cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]


def zoom(img, mask, z):
    H, W = mask.shape
    m = int(0.35 * max(z["w"], z["h"])) + 10
    x0, y0 = max(0, z["x"] - m), max(0, z["y"] - m)
    x1, y1 = min(W, z["x"] + z["w"] + m), min(H, z["y"] + z["h"] + m)
    crop = img[y0:y1, x0:x1].copy()
    c, _ = cv2.findContours(np.ascontiguousarray(mask[y0:y1, x0:x1]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, c, -1, tuple(int(v) for v in CORAIL), max(2, crop.shape[1] // 200))
    f = 480 / max(crop.shape[:2])
    return cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)


# ----------------------------------------------------------------------------- #
# Analyse appelée par app.py
# ----------------------------------------------------------------------------- #
# Les dernières images analysées restent en mémoire : changer le seuil ne relance pas le modèle.
CACHE = OrderedDict()
TAILLE_CACHE = 8


def analyser(modele, img=None, seuil=None, ident=None, max_zones=30):
    """Analyse une image (img) ou une image déjà analysée (ident) au seuil donné.
    Renvoie un dictionnaire JSON pour le navigateur, ou None si ident est inconnu."""
    seuil = seuil_defaut(modele) if seuil is None else float(seuil)
    ms = 0.0
    if img is not None:
        t0 = time.time()
        p = predict_proba(img, modele)
        ms = (time.time() - t0) * 1000
        ident = uuid.uuid4().hex
        CACHE[ident] = (img, p, ms)
        while len(CACHE) > TAILLE_CACHE:
            CACHE.popitem(last=False)
    elif ident in CACHE:
        img, p, ms = CACHE[ident]
        CACHE.move_to_end(ident)
    else:
        return None

    mask, zones = trouver_zones(p, seuil)
    H, W = mask.shape
    return {
        "id": ident,
        "seuil": seuil,
        "ms": round(ms),
        "nb_zones": len(zones),
        "surface": pct(mask.mean(), 1),
        "conf_max": pct(max((z["conf_max"] for z in zones), default=0)),
        "resultat": en_base64(dessiner(img, mask, zones)),
        "carte": en_base64(carte_chaleur(p), 85),
        "zones": [{
            "numero": i + 1,
            "position": z["pos"],
            "surface": pct(z["part"], 2),
            "confiance": pct(z["conf_moy"]),
            "conf_max": pct(z["conf_max"]),
            "niveau": niveau(z["conf_moy"]),
            "taille": f"{z['w']} × {z['h']} px",
            # cadre en proportion de l'image, pour la surbrillance côté navigateur
            "cadre": [z["x"] / W, z["y"] / H, z["w"] / W, z["h"] / H],
            "zoom": en_base64(zoom(img, mask, z), 88),
        } for i, z in enumerate(zones[:max_zones])],
    }

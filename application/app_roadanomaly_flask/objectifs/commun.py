"""Outils partagés par toutes les pages : chemins, lecture d'image, conversion pour le navigateur."""
import base64
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parent.parent
FORMATS_IMAGE = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def dossier_modele(numero):
    """Dossier du modèle d'un objectif : modeles/objectif_N/"""
    return RACINE / "modeles" / f"objectif_{numero}"


def dossier_exemples(numero):
    return RACINE / "exemples" / f"objectif_{numero}"


def exemples(numero):
    """Noms des images d'exemple d'un objectif (exemples/objectif_N/)."""
    d = dossier_exemples(numero)
    return sorted(p.name for p in d.glob("*") if p.suffix.lower() in FORMATS_IMAGE) if d.exists() else []


def lire_image(octets):
    """Octets d'un fichier image envoyé par le navigateur -> tableau RGB (H, W, 3) uint8."""
    img = cv2.imdecode(np.frombuffer(octets, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Ce fichier n'est pas une image lisible (formats acceptés : JPG, PNG, WEBP, BMP).")
    return np.ascontiguousarray(img[..., ::-1])


def reduire(img, cote_max=1600):
    """Réduit une très grande photo (le plus grand côté ramené à cote_max pixels)."""
    h, w = img.shape[:2]
    f = cote_max / max(h, w)
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img


def en_base64(img, qualite=90):
    """Tableau RGB -> texte « data:image/jpeg;base64,… » affichable directement dans une balise <img>."""
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, qualite])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def pct(v, d=0):
    """Pourcentage à la française : 0.0213 -> '2,1 %'."""
    return f"{v * 100:.{d}f} %".replace(".", ",")

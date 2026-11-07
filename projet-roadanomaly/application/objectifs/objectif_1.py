"""Objectif 1 — Segmentation des anomalies routières (DeepLabV3+).

Localise au pixel près les objets inattendus sur la route, les numérote, et permet
d'examiner chaque zone. Modèle : DeepLabV3+ / ResNet34 pré-entraîné ImageNet, entraîné
sur RoadAnomaly, exporté en ONNX (fichiers dans modeles/objectif_1/).
"""
import json
import time

import cv2
import gradio as gr
import numpy as np

from .commun import dossier_modele, entete, exemples, pct

INFO = {
    "bouton": "Segmentation",
    "titre": "Segmentation des anomalies routières",
    "membre": "Mon objectif",
    "description": ("Déposez une photo de route : le modèle repère au pixel près les objets inattendus "
                    "(animaux, rochers, cônes, pneus, obstacles) et les numérote. Cliquez sur une zone "
                    "du tableau pour l'examiner de près."),
}
PRET = True

DOSSIER = dossier_modele(1)
CFG, SESSION, MEAN, STD = None, None, None, None


def charger():
    """Charge le modèle ONNX et sa configuration (une seule fois)."""
    global CFG, SESSION, MEAN, STD
    import onnxruntime as ort
    CFG = json.load(open(DOSSIER / "config.json", encoding="utf-8"))
    SESSION = ort.InferenceSession(str(DOSSIER / "model.onnx"), providers=ort.get_available_providers())
    MEAN = np.array(CFG["mean"], np.float32)
    STD = np.array(CFG["std"], np.float32)


# ----------------------------------------------------------------------------- #
# Modèle et dessin
# ----------------------------------------------------------------------------- #
# Couleurs (RVB) utilisées sur les images
ROSE = np.array([214, 76, 120])        # remplissage des anomalies
ROSE_FONCE = (150, 40, 80)             # contours
CREME = (255, 240, 245)                # cadres et numéros
SURBRILLANCE = (255, 214, 0)           # zone sélectionnée


# ----------------------------------------------------------------------------- #
# Modèle
# ----------------------------------------------------------------------------- #
def predict_proba(img):
    """Image RGB (H, W, 3) uint8 de taille quelconque -> probabilité d'anomalie (H, W)."""
    H, W = img.shape[:2]
    w = CFG["work_width"]
    h = max(32, int(round(H * w / W)))
    x = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    x = np.pad(x, ((0, (-h) % 32), (0, (-w) % 32), (0, 0)))
    x = ((x.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    logits = SESSION.run(None, {"image": x})[0][0, 0, :h, :w]
    p = 1 / (1 + np.exp(-logits))
    return cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)


def position(cx, cy, W, H):
    horiz = "à gauche" if cx < W / 3 else ("au centre" if cx < 2 * W / 3 else "à droite")
    vert = "en haut" if cy < H / 3 else ("au milieu" if cy < 2 * H / 3 else "en bas")
    return f"{vert}, {horiz}"


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
        zones.append(dict(x=x, y=y, w=w, h=h, aire=a, part=a / (H * W),
                          conf_moy=float(probs.mean()), conf_max=float(probs.max()),
                          pos=position(cent[i][0], cent[i][1], W, H)))
    zones.sort(key=lambda z: -z["aire"])
    return mask, zones


# ----------------------------------------------------------------------------- #
# Dessin
# ----------------------------------------------------------------------------- #
def dessiner(img, mask, zones, choisie=None):
    vis = img.copy()
    zone = mask.astype(bool)
    vis[zone] = (0.45 * vis[zone] + 0.55 * ROSE).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape
    ep = max(2, W // 450)
    cv2.drawContours(vis, contours, -1, ROSE_FONCE, ep)
    if choisie is not None:                                   # atténue le reste de l'image
        z = zones[choisie]
        voile = (vis * 0.45).astype(np.uint8)
        voile[z["y"]:z["y"] + z["h"], z["x"]:z["x"] + z["w"]] = vis[z["y"]:z["y"] + z["h"], z["x"]:z["x"] + z["w"]]
        vis = voile
    taille = max(0.5, W / 1400)
    for i, z in enumerate(zones):
        sel = i == choisie
        coul = SURBRILLANCE if sel else CREME
        cv2.rectangle(vis, (z["x"], z["y"]), (z["x"] + z["w"], z["y"] + z["h"]), coul, ep * (2 if sel else 1))
        txt = str(i + 1)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, taille, max(1, ep))
        y0 = max(th + 8, z["y"])
        cv2.rectangle(vis, (z["x"], y0 - th - 8), (z["x"] + tw + 10, y0), coul, -1)
        cv2.putText(vis, txt, (z["x"] + 5, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, taille, (60, 30, 40), max(1, ep),
                    cv2.LINE_AA)
    return vis


def carte_chaleur(p):
    return cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)[..., ::-1]


def zoom(img, mask, z):
    H, W = mask.shape
    m = int(0.35 * max(z["w"], z["h"])) + 10
    x0, y0 = max(0, z["x"] - m), max(0, z["y"] - m)
    x1, y1 = min(W, z["x"] + z["w"] + m), min(H, z["y"] + z["h"] + m)
    crop = img[y0:y1, x0:x1].copy()
    c, _ = cv2.findContours(mask[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, c, -1, (230, 60, 110), max(2, crop.shape[1] // 200))
    f = 520 / max(crop.shape[:2])
    return cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)


# ----------------------------------------------------------------------------- #
# Textes
# ----------------------------------------------------------------------------- #
def niveau(c):
    return "élevée" if c >= 0.8 else ("moyenne" if c >= 0.5 else "faible")


def carte_resume(zones, mask, ms, seuil):
    if not zones:
        return (f"<div class='resume vide'><div class='gros'>Aucune anomalie</div>"
                f"<div class='petit'>au seuil {seuil:.2f} · analyse en {ms:.0f} ms</div></div>")
    return ("<div class='resume'>"
            f"<div class='stat'><div class='gros'>{len(zones)}</div><div class='petit'>zone{'s' if len(zones) > 1 else ''} détectée{'s' if len(zones) > 1 else ''}</div></div>"
            f"<div class='stat'><div class='gros'>{pct(mask.mean(), 1)}</div><div class='petit'>de l'image</div></div>"
            f"<div class='stat'><div class='gros'>{pct(max(z['conf_max'] for z in zones))}</div><div class='petit'>confiance max.</div></div>"
            f"<div class='stat'><div class='gros'>{ms:.0f} ms</div><div class='petit'>temps d'analyse</div></div>"
            "</div>")


def tableau(zones):
    return [[i + 1, z["pos"], pct(z["part"], 2), pct(z["conf_moy"]), niveau(z["conf_moy"])]
            for i, z in enumerate(zones)]


def details(z, i):
    return (f"#### Zone {i + 1}\n"
            f"- **Position :** {z['pos']}\n"
            f"- **Taille :** {z['w']} × {z['h']} px, soit {pct(z['part'], 2)} de l'image\n"
            f"- **Confiance moyenne :** {pct(z['conf_moy'])} ({niveau(z['conf_moy'])}) · maximale : {pct(z['conf_max'])}")


# ----------------------------------------------------------------------------- #
# Actions de l'interface
# ----------------------------------------------------------------------------- #
VIDE = (None, None, "<div class='resume vide'><div class='petit'>Déposez une image puis cliquez sur « Analyser ».</div></div>",
        [], None, None, "*Cliquez sur une ligne du tableau pour agrandir la zone.*")


def analyser(img, seuil):
    if img is None:
        return VIDE
    img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
    t0 = time.time()
    p = predict_proba(img)
    ms = (time.time() - t0) * 1000
    mask, zones = trouver_zones(p, seuil)
    etat = {"img": img, "mask": mask, "zones": zones}
    aide = ("*Cliquez sur une ligne du tableau pour agrandir la zone.*" if zones
            else "*Aucune zone au-dessus du seuil : essayez de le baisser.*")
    return (dessiner(img, mask, zones), carte_chaleur(p), carte_resume(zones, mask, ms, seuil),
            tableau(zones), etat, None, aide)


def choisir_zone(etat, evt: gr.SelectData):
    if not etat or not etat["zones"]:
        return gr.update(), None, "*Aucune zone à afficher.*"
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    idx = int(idx)
    if idx >= len(etat["zones"]):
        return gr.update(), None, ""
    z = etat["zones"][idx]
    return dessiner(etat["img"], etat["mask"], etat["zones"], choisie=idx), zoom(etat["img"], etat["mask"], z), details(z, idx)


# ----------------------------------------------------------------------------- #
# Page
# ----------------------------------------------------------------------------- #
def construire(numero):
    manquants = [f for f in ("model.onnx", "config.json") if not (DOSSIER / f).exists()]
    if manquants:
        gr.HTML(entete(numero, INFO, "erreur"))
        gr.HTML("<div class='carte-attente erreur'><h3>Modèle introuvable</h3>"
                f"<p>Copiez <code>{'</code> et <code>'.join(manquants)}</code> dans le dossier "
                "<code>modeles/objectif_1/</code>, puis relancez <code>python app.py</code>.</p></div>")
        return
    charger()
    chips = [CFG["label"].split(". ", 1)[-1], f"AP {CFG['cv_AP']:.3f}",
             f"IoU {CFG['cv_IoU_mean']:.3f} ± {CFG['cv_IoU_std']:.3f}", "Validation croisée 5 plis"]
    gr.HTML(entete(numero, INFO, "pret", chips))
    etat = gr.State()

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            entree = gr.Image(type="numpy", label="Image à analyser", height=380)
            seuil = gr.Slider(0.05, 0.95, value=round(CFG["threshold"], 2), step=0.01, label="Seuil de décision",
                              info="Plus bas : plus de détections, mais plus de fausses alarmes.")
            with gr.Row():
                effacer = gr.Button("Effacer", variant="secondary")
                bouton = gr.Button("Analyser", variant="primary")
            ex = exemples(numero)
            if ex:
                gr.Examples(examples=[[p] for p in ex], inputs=[entree],
                            label="Exemples (cliquez pour charger)", examples_per_page=8)

        with gr.Column(scale=7):
            resume = gr.HTML(VIDE[2])
            with gr.Tabs():
                with gr.Tab("Anomalies détectées"):
                    sortie = gr.Image(label="Résultat", height=420, interactive=False)
                with gr.Tab("Carte de probabilité"):
                    carte = gr.Image(label="Probabilité d'anomalie (clair = élevée)", height=420, interactive=False)

    with gr.Row(equal_height=False):
        with gr.Column(scale=7):
            zones = gr.Dataframe(headers=["N°", "Position", "Surface", "Confiance", "Niveau"],
                                 datatype=["number", "str", "str", "str", "str"], interactive=False,
                                 label="Zones détectées (cliquez sur une ligne)", elem_id="tableau-zones")
        with gr.Column(scale=5):
            loupe = gr.Image(label="Zone agrandie", height=300, interactive=False)
            info_zone = gr.Markdown(VIDE[6])

    sorties = [sortie, carte, resume, zones, etat, loupe, info_zone]
    bouton.click(analyser, [entree, seuil], sorties)
    entree.change(lambda im, s: analyser(im, s) if im is not None else VIDE, [entree, seuil], sorties)
    seuil.release(analyser, [entree, seuil], sorties)
    effacer.click(lambda: (None, *VIDE), None, [entree, *sorties])
    zones.select(choisir_zone, [etat], [sortie, loupe, info_zone])

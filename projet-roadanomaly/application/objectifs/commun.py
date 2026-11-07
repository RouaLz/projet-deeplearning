"""Outils partagés par toutes les pages : chemins, en-têtes, page générique."""
import html
import time
import traceback
from pathlib import Path

import gradio as gr
import numpy as np

RACINE = Path(__file__).resolve().parent.parent
FORMATS_IMAGE = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def dossier_modele(numero):
    return RACINE / "modeles" / f"objectif_{numero}"


def exemples(numero):
    d = RACINE / "exemples" / f"objectif_{numero}"
    return sorted(str(p) for p in d.glob("*") if p.suffix.lower() in FORMATS_IMAGE) if d.exists() else []


def pct(v, d=0):
    """Pourcentage à la française : 0.0213 -> '2,1 %'."""
    return f"{v * 100:.{d}f} %".replace(".", ",")


def entete(numero, info, statut=None, chips=()):
    """Carte d'en-tête d'une page : numéro, titre, membre, description, statut, pastilles."""
    e = html.escape
    badge = ""
    if statut == "pret":
        badge = "<span class='badge ok'>Opérationnel</span>"
    elif statut == "attente":
        badge = "<span class='badge attente'>En attente d'intégration</span>"
    elif statut == "erreur":
        badge = "<span class='badge erreur'>Erreur au chargement</span>"
    pastilles = "".join(f"<span class='pastille'>{e(c)}</span>" for c in chips)
    return f"""
<div class="entete-objectif">
  <div class="numero">{numero}</div>
  <div class="texte">
    <div class="ligne-titre"><h2>{e(info.get("titre", f"Objectif {numero}"))}</h2>{badge}</div>
    <div class="membre">{e(info.get("membre", ""))}</div>
    <p>{e(info.get("description", ""))}</p>
    {f"<div class='pastilles'>{pastilles}</div>" if pastilles else ""}
  </div>
</div>"""


def page_attente(numero, info):
    """Page affichée tant que le code d'un objectif n'est pas encore intégré."""
    gr.HTML(entete(numero, info, "attente"))
    gr.HTML(f"""
<div class="carte-attente">
  <h3>Cette page attend le code de {html.escape(info.get("membre", "ce membre"))}</h3>
  <p>Pour activer cet objectif, dans le fichier <code>objectifs/objectif_{numero}.py</code> :</p>
  <ol>
    <li>compléter <code>INFO</code> (titre, membre, description, texte du bouton) ;</li>
    <li>écrire <code>charger()</code> : charger le modèle depuis <code>modeles/objectif_{numero}/</code> ;</li>
    <li>écrire <code>predire(image, modele)</code> : renvoyer une image résultat et un texte ;</li>
    <li>passer <code>PRET = True</code>, puis relancer <code>python app.py</code>.</li>
  </ol>
  <p>Images d'exemple (facultatif) : les déposer dans <code>exemples/objectif_{numero}/</code>.
  Le détail est dans <code>GUIDE_INTEGRATION.md</code>.</p>
</div>""")


def page_erreur(numero, info, erreur):
    gr.HTML(entete(numero, info, "erreur"))
    gr.HTML("<div class='carte-attente erreur'><h3>Le module n'a pas pu être chargé</h3>"
            f"<pre>{html.escape(erreur)}</pre></div>")


def page_generique(numero, module):
    """Interface standard : image -> predire() -> image résultat + texte.
    Utilisée pour tout objectif qui ne définit pas sa propre fonction construire()."""
    info = module.INFO
    gr.HTML(entete(numero, info, "pret", info.get("pastilles", ())))
    try:
        modele = module.charger()
    except Exception:
        gr.HTML("<div class='carte-attente erreur'><h3>Échec de charger()</h3>"
                f"<pre>{html.escape(traceback.format_exc(limit=3))}</pre></div>")
        return

    def lancer(img):
        if img is None:
            return None, "*Déposez une image puis cliquez sur « Analyser ».*"
        img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
        t0 = time.time()
        try:
            res_img, texte = module.predire(img, modele)
        except Exception:
            return None, f"**Erreur dans predire() :**\n```\n{traceback.format_exc(limit=3)}\n```"
        return res_img, f"{texte}\n\n*Analyse en {(time.time() - t0) * 1000:.0f} ms.*"

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            entree = gr.Image(type="numpy", label="Image à analyser", height=380)
            with gr.Row():
                effacer = gr.Button("Effacer", variant="secondary")
                bouton = gr.Button("Analyser", variant="primary")
            ex = exemples(numero)
            if ex:
                gr.Examples(examples=[[p] for p in ex], inputs=[entree],
                            label="Exemples (cliquez pour charger)", examples_per_page=8)
        with gr.Column(scale=7):
            sortie = gr.Image(label=info.get("libelle_resultat", "Résultat"), height=420, interactive=False)
            texte = gr.Markdown("*Déposez une image puis cliquez sur « Analyser ».*")
    bouton.click(lancer, entree, [sortie, texte])
    entree.change(lancer, entree, [sortie, texte])
    effacer.click(lambda: (None, None, "*Déposez une image puis cliquez sur « Analyser ».*"), None,
                  [entree, sortie, texte])

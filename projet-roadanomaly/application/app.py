"""Détection d'anomalies routières — application du projet de groupe (version 3).

Une barre de boutons en haut permet de passer d'un objectif à l'autre, dans la même fenêtre.
Chaque objectif est un fichier du dossier « objectifs » (objectif_1.py … objectif_6.py).

Lancement :  python app.py   puis ouvrir http://127.0.0.1:7860
"""
import importlib
import traceback

import gradio as gr

from objectifs import MODULES
from objectifs.commun import page_attente, page_erreur, page_generique

# ----------------------------------------------------------------------------- #
# Thème et style « dusty rose »
# ----------------------------------------------------------------------------- #
DUSTY_ROSE = gr.themes.Color(c50="#FBF4F5", c100="#F5E6E9", c200="#EBCDD3", c300="#DDB0B9", c400="#CC929E",
                             c500="#B97A88", c600="#A06473", c700="#82505D", c800="#673F4A", c900="#4F313A",
                             c950="#3A242B", name="dusty_rose")
THEME = gr.themes.Soft(
    primary_hue=DUSTY_ROSE, secondary_hue=DUSTY_ROSE, neutral_hue="stone", radius_size="lg",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#FAF5F6",
    block_background_fill="#FFFFFF",
    block_border_width="0px",
    block_shadow="0 6px 24px rgba(130, 80, 93, 0.10)",
    block_label_background_fill="#F5E6E9",
    block_label_text_color="#82505D",
    button_primary_background_fill="linear-gradient(135deg, #C48B9A 0%, #A06473 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #B97A88 0%, #82505D 100%)",
    button_primary_text_color="#FFFFFF",
    slider_color="#B97A88",
)

CSS = """
.gradio-container {max-width: 1280px !important; width: 100% !important; margin: auto !important;}
#banniere {background: linear-gradient(120deg, #D9A5B3 0%, #B97A88 55%, #8E5A69 100%);
           border-radius: 22px; padding: 26px 34px; color: #fff; box-shadow: 0 10px 30px rgba(130,80,93,.25);}
#banniere h1 {color: #fff; font-size: 2rem; margin: 0 0 6px 0; font-weight: 700; letter-spacing: -0.5px;}
#banniere p {color: #FDF2F5; margin: 0; font-size: 1rem; max-width: 820px;}

#barre-nav {background: #FFFFFF; border-radius: 18px; padding: 10px; gap: 8px; flex-wrap: wrap;
            box-shadow: 0 6px 24px rgba(130, 80, 93, 0.10);}
#barre-nav button.nav-btn {border-radius: 999px !important; min-width: 0 !important; flex: 1 1 170px;
                           font-weight: 600; padding: 10px 14px;}
#barre-nav button.nav-btn.secondary {background: #FBF4F5 !important; color: #82505D !important; border: 0 !important;}
#barre-nav button.nav-btn.secondary:hover {background: #F5E6E9 !important;}

.entete-objectif {display: flex; gap: 18px; align-items: flex-start; background: #FFFFFF; border-radius: 20px;
                  padding: 20px 24px; box-shadow: 0 6px 24px rgba(130, 80, 93, 0.10);}
.entete-objectif .numero {width: 52px; height: 52px; border-radius: 16px; flex-shrink: 0; color: #fff;
                          background: linear-gradient(135deg, #D9A5B3, #A06473); font-weight: 700; font-size: 1.4rem;
                          display: flex; align-items: center; justify-content: center;}
.entete-objectif .ligne-titre {display: flex; align-items: center; gap: 12px; flex-wrap: wrap;}
.entete-objectif h2 {margin: 0; color: #4F313A; font-size: 1.35rem;}
.entete-objectif .membre {color: #A06473; font-weight: 600; font-size: .9rem; margin-top: 2px;}
.entete-objectif p {color: #6B5A5F; margin: 8px 0 0 0;}
.badge {padding: 4px 12px; border-radius: 999px; font-size: .78rem; font-weight: 600;}
.badge.ok {background: #E3F1E7; color: #2F6B45;}
.badge.attente {background: #FBEFD9; color: #8A5A12;}
.badge.erreur {background: #FBE3E3; color: #9B2C2C;}
.pastilles {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px;}
.pastille {background: #F5E6E9; color: #82505D; padding: 4px 12px; border-radius: 999px; font-size: .8rem;}

.carte-attente {background: #FFFFFF; border: 2px dashed #DDB0B9; border-radius: 20px; padding: 22px 28px; color: #4F313A;}
.carte-attente h3 {margin-top: 0; color: #82505D;}
.carte-attente code {background: #F5E6E9; color: #673F4A; padding: 1px 6px; border-radius: 6px;}
.carte-attente.erreur {border-color: #E8A9A9;}
.carte-attente pre {white-space: pre-wrap; font-size: .8rem; background: #FBF4F5; padding: 12px; border-radius: 10px;}

.resume {display: flex; gap: 12px; flex-wrap: wrap;}
.resume .stat {flex: 1; min-width: 120px; background: #FBF4F5; border-radius: 16px; padding: 14px 16px; text-align: center;}
.resume .gros {font-size: 1.6rem; font-weight: 700; color: #82505D; line-height: 1.2;}
.resume .petit {font-size: .82rem; color: #8B7378;}
.resume.vide {background: #FBF4F5; border-radius: 16px; padding: 16px; text-align: center; display: block;}
#tableau-zones .table-wrap, #tableau-zones .table-wrap * {font-family: var(--font) !important;}
#tableau-zones .body-cell, #tableau-zones td {cursor: pointer;}
#tableau-zones tr:hover .body-cell, #tableau-zones tr:hover td {background: #F5E6E9 !important;}
#avertissement {font-size: .82rem; color: #8B7378; text-align: center;}
footer {display: none !important;}
"""

# l'interface est conçue en clair : on désactive le mode sombre automatique du navigateur
FORCER_CLAIR = """() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u); }
}"""

BANNIERE = """
<div id="banniere">
  <h1>Détection d'anomalies sur la route</h1>
  <p>Projet de groupe de deep learning sur le dataset RoadAnomaly. Choisissez un objectif dans la barre
  ci-dessous : chaque bouton ouvre l'interface correspondante.</p>
</div>
"""

GRADIO6 = int(gr.__version__.split(".")[0]) >= 6
STYLE = dict(theme=THEME, css=CSS, js=FORCER_CLAIR)


# ----------------------------------------------------------------------------- #
# Chargement des objectifs
# ----------------------------------------------------------------------------- #
def importer(nom):
    try:
        return importlib.import_module(f"objectifs.{nom}"), None
    except Exception:
        return None, traceback.format_exc(limit=4)


OBJECTIFS = []                        # (numéro, module ou None, erreur ou None)
for i, nom in enumerate(MODULES, start=1):
    module, erreur = importer(nom)
    OBJECTIFS.append((i, module, erreur))


def libelle(numero, module):
    texte = module.INFO.get("bouton", f"Objectif {numero}") if module else f"Objectif {numero}"
    return f"{numero} · {texte}"


def construire_page(numero, module, erreur):
    if module is None:
        page_erreur(numero, {"titre": f"Objectif {numero}"}, erreur)
    elif hasattr(module, "construire"):
        try:
            module.construire(numero)
        except Exception:
            page_erreur(numero, module.INFO, traceback.format_exc(limit=4))
    elif getattr(module, "PRET", False):
        page_generique(numero, module)
    else:
        page_attente(numero, module.INFO)


# ----------------------------------------------------------------------------- #
# Interface : bannière, barre de navigation, une page par objectif
# ----------------------------------------------------------------------------- #
with gr.Blocks(title="Détection d'anomalies routières", fill_width=True, **({} if GRADIO6 else STYLE)) as demo:
    gr.HTML(BANNIERE)

    with gr.Row(elem_id="barre-nav"):
        boutons = [gr.Button(libelle(n, m), variant="primary" if n == 1 else "secondary",
                             elem_classes="nav-btn", size="sm") for n, m, _ in OBJECTIFS]

    pages = []
    for n, m, err in OBJECTIFS:
        with gr.Column(visible=(n == 1)) as page:
            construire_page(n, m, err)
        pages.append(page)

    gr.Markdown("Projet étudiant de deep learning, entraîné sur peu d'images : les résultats sur des scènes "
                "très différentes peuvent être mauvais. Ce n'est pas un système de sécurité.", elem_id="avertissement")

    def afficher(choix):
        return ([gr.update(visible=(j == choix)) for j in range(len(pages))]
                + [gr.update(variant="primary" if j == choix else "secondary") for j in range(len(boutons))])

    for i, b in enumerate(boutons):
        b.click(lambda i=i: afficher(i), None, pages + boutons)


def lancer(**kwargs):
    """Lance l'application (gère la différence entre Gradio 5 et Gradio 6 pour le thème)."""
    if GRADIO6:
        kwargs = {**STYLE, **kwargs}
    return demo.launch(**kwargs)


if __name__ == "__main__":
    lancer()

"""Détection d'anomalies routières — application web du projet de groupe (version Flask).

Une barre de boutons en haut permet de passer d'un objectif à l'autre, dans la même fenêtre.
Chaque objectif est un fichier du dossier « objectifs » (objectif_1.py … objectif_6.py).

Lancement :  python app.py   puis ouvrir http://127.0.0.1:5000
Arrêt     :  Ctrl + C dans le terminal
"""
import importlib
import time
import traceback

import markdown
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for

from objectifs import MODULES
from objectifs.commun import dossier_exemples, dossier_modele, en_base64, exemples, lire_image, reduire

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024          # images de 25 Mo au plus

# Une couleur par objectif (bouton de navigation, numéro, accents de la page)
COULEURS = ["#0D9488", "#7C3AED", "#D97706", "#E11D48", "#2563EB", "#16A34A"]


# ----------------------------------------------------------------------------- #
# Chargement des objectifs (une seule fois, au démarrage)
# ----------------------------------------------------------------------------- #
class Objectif:
    def __init__(self, numero, nom):
        self.numero = numero
        self.couleur = COULEURS[(numero - 1) % len(COULEURS)]
        self.module, self.modele, self.erreur, self.manquants = None, None, None, []
        self.info = {"titre": f"Objectif {numero}", "bouton": f"Objectif {numero}"}
        try:
            self.module = importlib.import_module(f"objectifs.{nom}")
            self.info = {**self.info, **self.module.INFO}
        except Exception:
            self.erreur = traceback.format_exc(limit=4)
            return
        if not getattr(self.module, "PRET", False):
            return
        dossier = dossier_modele(numero)
        self.manquants = [f for f in getattr(self.module, "FICHIERS_REQUIS", ()) if not (dossier / f).exists()]
        if self.manquants:
            return
        try:
            print(f"Chargement de l'objectif {numero}…")
            self.modele = self.module.charger()
        except Exception:
            self.erreur = traceback.format_exc(limit=4)

    @property
    def statut(self):
        if self.erreur or self.manquants:
            return "erreur"
        return "pret" if getattr(self.module, "PRET", False) else "attente"

    @property
    def pastilles(self):
        if self.statut == "pret" and hasattr(self.module, "pastilles"):
            return self.module.pastilles(self.modele)
        return self.info.get("pastilles", [])


OBJECTIFS = {i: Objectif(i, nom) for i, nom in enumerate(MODULES, start=1)}


def objectif_ou_404(numero):
    if numero not in OBJECTIFS:
        abort(404)
    return OBJECTIFS[numero]


# ----------------------------------------------------------------------------- #
# Pages
# ----------------------------------------------------------------------------- #
@app.route("/")
def accueil():
    return redirect(url_for("page", numero=1))


@app.route("/objectif/<int:numero>")
def page(numero):
    obj = objectif_ou_404(numero)
    contexte = dict(obj=obj, objectifs=OBJECTIFS.values(), exemples=exemples(numero))
    if obj.statut == "erreur":
        return render_template("erreur.html", **contexte)
    if obj.statut == "attente":
        return render_template("attente.html", **contexte)
    gabarit = getattr(obj.module, "GABARIT", "generique.html")
    if hasattr(obj.module, "seuil_defaut"):
        contexte["seuil"] = obj.module.seuil_defaut(obj.modele)
    return render_template(gabarit, **contexte)


@app.route("/exemples/<int:numero>/<path:nom>")
def image_exemple(numero, nom):
    return send_from_directory(dossier_exemples(numero), nom)


# ----------------------------------------------------------------------------- #
# API : analyse d'une image
# ----------------------------------------------------------------------------- #
def image_recue(numero):
    """L'image envoyée : un fichier (champ « image ») ou le nom d'un exemple (champ « exemple »)."""
    if "image" in request.files and request.files["image"].filename:
        return reduire(lire_image(request.files["image"].read()))
    nom = request.form.get("exemple")
    if nom and nom in exemples(numero):
        return reduire(lire_image((dossier_exemples(numero) / nom).read_bytes()))
    return None


@app.route("/api/objectif/<int:numero>/analyser", methods=["POST"])
def api_analyser(numero):
    obj = objectif_ou_404(numero)
    if obj.statut != "pret":
        return jsonify(erreur="Cet objectif n'est pas encore disponible."), 400
    try:
        img = image_recue(numero)
        ident = request.form.get("id")

        # Page personnalisée (objectif 1) : le module fournit sa propre fonction analyser()
        if hasattr(obj.module, "analyser"):
            if img is None and not ident:
                return jsonify(erreur="Aucune image reçue."), 400
            res = obj.module.analyser(obj.modele, img=img, seuil=request.form.get("seuil"),
                                      ident=None if img is not None else ident)
            if res is None:                      # image plus en mémoire : le navigateur la renverra
                return jsonify(erreur="expire"), 410
            return jsonify(res)

        # Page standard : predire(image, modele) -> (image_resultat, texte)
        if img is None:
            return jsonify(erreur="Aucune image reçue."), 400
        t0 = time.time()
        image_res, texte = obj.module.predire(img, obj.modele)
        return jsonify(image=en_base64(image_res), texte=markdown.markdown(texte or ""),
                       ms=round((time.time() - t0) * 1000))
    except ValueError as e:
        return jsonify(erreur=str(e)), 400
    except Exception:
        traceback.print_exc()
        return jsonify(erreur="Erreur pendant l'analyse :\n" + traceback.format_exc(limit=3)), 500


@app.errorhandler(413)
def trop_gros(_):
    return jsonify(erreur="Image trop lourde (25 Mo au maximum)."), 413


if __name__ == "__main__":
    print("\nApplication prête : ouvrez http://127.0.0.1:5000 dans le navigateur (Ctrl + C pour arrêter)\n")
    app.run(host="127.0.0.1", port=5000, debug=False)

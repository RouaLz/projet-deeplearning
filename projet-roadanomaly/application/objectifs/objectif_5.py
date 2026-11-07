"""Objectif 5 — à compléter par le membre responsable.

Ce fichier est la page n° 5 de l'application. Pour l'activer :
  1. compléter INFO ci-dessous ;
  2. déposer le modèle (et ses fichiers) dans modeles/objectif_5/ ;
  3. écrire charger() et predire() ;
  4. passer PRET = True, puis relancer « python app.py ».
Les images d'exemple (facultatif) vont dans exemples/objectif_5/.
Voir GUIDE_INTEGRATION.md pour des exemples complets (classification, détection, etc.).
"""
import numpy as np

from .commun import dossier_modele

INFO = {
    "bouton": "Objectif 5",                       # texte du bouton dans la barre de navigation
    "titre": "Objectif 5",                        # titre de la page
    "membre": "Membre 5",                         # nom du membre responsable
    "description": "Description de l'objectif à compléter.",
    "libelle_resultat": "Résultat",              # titre du cadre de l'image résultat
    # "pastilles": ["Modèle : …", "Score : …"],    # facultatif : petites étiquettes sous le titre
}
PRET = False          # passer à True quand charger() et predire() sont écrits

DOSSIER = dossier_modele(5)                     # = modeles/objectif_5/


def charger():
    """Appelée une seule fois au démarrage : charger et renvoyer le modèle.
    Exemple : return onnxruntime.InferenceSession(str(DOSSIER / "model.onnx"))"""
    return None


def predire(image, modele):
    """image : photo RGB (tableau numpy hauteur x largeur x 3, valeurs 0-255).
    modele : ce que charger() a renvoyé.
    Doit renvoyer (image_resultat, texte) :
      - image_resultat : tableau numpy RGB à afficher (ou None) ;
      - texte : résultat en Markdown, par exemple "**Classe prédite :** animal (92 %)".
    """
    raise NotImplementedError("predire() n'est pas encore écrite")


# Facultatif : pour une interface entièrement personnalisée, définir
#   def construire(numero): ...   (composants Gradio)
# Elle remplace alors la page standard « image -> predire() -> résultat ».

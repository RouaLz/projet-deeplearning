# Application — Détection d'anomalies routières (version Flask)

Une application web en **Flask** avec une barre de 6 boutons en haut de la page. Chaque bouton ouvre
l'interface d'un objectif, dans la même fenêtre. Le bouton 1 est la segmentation des anomalies
(DeepLabV3+). Les boutons 2 à 6 attendent le travail des autres membres (voir `GUIDE_INTEGRATION.md`).

## Installation (une seule fois)

1. Copiez **`model.onnx`** et **`config.json`** (produits par le notebook 05, les mêmes que pour
   l'application Gradio) dans le dossier **`modeles/objectif_1/`**.
2. Ouvrez un terminal dans ce dossier et tapez :

   ```
   pip install -r requirements.txt
   ```

## Lancement

```
python app.py
```

puis ouvrez **http://127.0.0.1:5000** dans le navigateur. Pour arrêter : **Ctrl + C** dans le terminal.

## Utilisation (objectif 1)

- Déposez une photo (clic, glisser-déposer ou Ctrl + V) ou cliquez sur un exemple : l'analyse se lance.
- Le curseur **Seuil** change la sensibilité ; l'image n'est pas réanalysée, seul le seuil est réappliqué.
- Cliquez sur une **ligne du tableau** ou directement sur une **zone de l'image** : elle est mise en
  surbrillance et agrandie à droite.

## Organisation des fichiers

```
app.py                  serveur Flask : pages, navigation, API d'analyse
objectifs/              un fichier Python par objectif (+ commun.py, outils partagés)
templates/              pages HTML (Jinja) : base, objectif 1, page standard, attente, erreur
static/                 style.css (thème) et scripts JavaScript
modeles/objectif_N/     modèle de chaque objectif
exemples/objectif_N/    images d'exemple de chaque objectif
```

## Comment ça marche

1. Au démarrage, `app.py` importe chaque fichier `objectifs/objectif_N.py` et appelle `charger()`
   une seule fois (le modèle reste en mémoire).
2. `/objectif/N` renvoie la page HTML de l'objectif N.
3. Quand on clique sur « Analyser », le navigateur envoie l'image à `/api/objectif/N/analyser`
   (requête POST). Le serveur lance le modèle et répond en JSON (images encodées en base64, chiffres,
   liste des zones). Le JavaScript met la page à jour sans la recharger.

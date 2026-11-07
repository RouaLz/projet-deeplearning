# Application — Détection d'anomalies routières (version 3, projet de groupe)

Une barre de 6 boutons en haut de la page : chaque bouton ouvre l'interface d'un objectif, dans
la même fenêtre. Le bouton 1 est la segmentation des anomalies (DeepLabV3+). Les boutons 2 à 6
attendent le travail des autres membres (voir `GUIDE_INTEGRATION.md`).

## Installation (une seule fois)

1. Copiez **`model.onnx`** et **`config.json`** (ceux de votre première application) dans le
   dossier **`modeles/objectif_1/`**.
2. Ouvrez un terminal dans ce dossier et tapez :

   ```
   pip install -r requirements.txt
   ```

## Lancement

```
python app.py
```

puis ouvrez **http://127.0.0.1:7860** dans le navigateur. Pour arrêter : **Ctrl + C** dans le
terminal.

## Intégrer le travail d'un membre

Voir **`GUIDE_INTEGRATION.md`** : chaque membre remplit uniquement le fichier
`objectifs/objectif_N.py` et dépose son modèle dans `modeles/objectif_N/`.

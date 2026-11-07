# Guide d'intégration — ajouter son objectif à l'application

L'application a une barre de 6 boutons. Le bouton 1 est la **segmentation des anomalies** (déjà
fait). Les boutons 2 à 6 attendent le travail des autres membres : chacun remplit **un seul
fichier**, `objectifs/objectif_N.py`, où N est son numéro.

```
app_roadanomaly_v3/
├── app.py                     ← ne pas modifier (barre de navigation, thème)
├── objectifs/
│   ├── commun.py              ← outils partagés (ne pas modifier)
│   ├── objectif_1.py          ← segmentation (fait)
│   ├── objectif_2.py          ← membre 2
│   ├── …
│   └── objectif_6.py          ← membre 6
├── modeles/objectif_N/        ← le modèle du membre N (poids, configuration)
├── exemples/objectif_N/       ← ses images d'exemple (facultatif)
└── requirements.txt           ← ajouter ses bibliothèques
```

## Les 4 étapes

1. **Compléter `INFO`** : texte du bouton (court : 1 à 2 mots), titre de la page, nom du membre,
   description en une ou deux phrases.
2. **Déposer le modèle** dans `modeles/objectif_N/`.
3. **Écrire deux fonctions** :
   - `charger()` : appelée **une seule fois** au démarrage ; elle charge le modèle et le renvoie ;
   - `predire(image, modele)` : reçoit une photo RGB (tableau numpy, valeurs 0 à 255) et renvoie
     `(image_resultat, texte)`. `image_resultat` est l'image à afficher (ou `None`) et `texte`
     le résultat en Markdown.
4. **Passer `PRET = True`**, ajouter ses bibliothèques à `requirements.txt`, relancer
   `python app.py`.

La page standard (image → bouton « Analyser » → image résultat + texte + exemples) est alors
construite automatiquement, avec le même design que la page 1.

Tant que `PRET = False`, la page affiche « En attente d'intégration ». Si le fichier contient une
erreur, seule cette page affiche l'erreur : les autres pages continuent de fonctionner.

## Exemple 1 : classification (modèle PyTorch)

```python
import torch, torchvision, cv2, numpy as np
from .commun import dossier_modele

INFO = {"bouton": "Classification", "titre": "Type d'anomalie", "membre": "Prénom Nom",
        "description": "Reconnaît le type d'anomalie présent sur la photo."}
PRET = True
DOSSIER = dossier_modele(2)
CLASSES = ["animal", "obstacle", "cône", "pneu", "véhicule"]

def charger():
    m = torchvision.models.resnet18(num_classes=len(CLASSES))
    m.load_state_dict(torch.load(DOSSIER / "model.pt", map_location="cpu"))
    return m.eval()

def predire(image, modele):
    x = cv2.resize(image, (224, 224)).astype(np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.from_numpy(x.transpose(2, 0, 1)).float()[None]
    with torch.no_grad():
        p = modele(x).softmax(1)[0].numpy()
    lignes = "\n".join(f"- {c} : {v:.0%}" for c, v in sorted(zip(CLASSES, p), key=lambda t: -t[1]))
    return image, f"**Classe prédite : {CLASSES[p.argmax()]}** ({p.max():.0%})\n\n{lignes}"
```

## Exemple 2 : détection par boîtes (YOLO, bibliothèque ultralytics)

```python
from .commun import dossier_modele

INFO = {"bouton": "Détection", "titre": "Détection par boîtes englobantes", "membre": "Prénom Nom",
        "description": "Encadre chaque obstacle sur la route."}
PRET = True
DOSSIER = dossier_modele(3)

def charger():
    from ultralytics import YOLO
    return YOLO(str(DOSSIER / "best.pt"))

def predire(image, modele):
    r = modele(image, verbose=False)[0]
    vis = r.plot()[..., ::-1]                      # ultralytics dessine en BGR -> RGB
    return vis, f"**{len(r.boxes)} objet(s) détecté(s)**"
```

## Pour une interface différente

Si la page standard ne convient pas (plusieurs images en entrée, une vidéo, des curseurs…),
définir dans son fichier une fonction `construire(numero)` qui crée ses propres composants
Gradio. Elle remplace alors la page standard. Voir `objectif_1.py`, qui le fait pour le tableau
des zones cliquables. Utiliser `entete(numero, INFO, "pret")` (de `commun.py`) pour garder le
même en-tête que les autres pages.

## Avec Git

- Chaque membre ne modifie que **ses** fichiers : `objectifs/objectif_N.py`, `modeles/objectif_N/`,
  `exemples/objectif_N/`, et ajoute ses lignes à `requirements.txt`. Il n'y a donc pas de conflit.
- GitHub refuse les fichiers de plus de **100 Mo**. Pour un modèle plus lourd, utiliser Git LFS
  (`git lfs track "*.onnx" "*.pt"`) ou partager le modèle à part (lien de téléchargement) et le
  déposer à la main dans `modeles/objectif_N/`.
- Les images du dataset RoadAnomaly sont réservées à la recherche : ne pas les publier dans un
  dépôt public.

# Détection d'anomalies routières — RoadAnomaly

Projet de groupe du module **Deep Learning** : détecter les objets inattendus et dangereux sur la
route (animaux, rochers, cônes, pneus perdus, véhicules inhabituels) à partir du dataset
[RoadAnomaly](https://www.epfl.ch/labs/cvlab/data/road-anomaly/) (Lis et al., ICCV 2019).

Le dépôt contient les notebooks (exécutables sur Google Colab), le code partagé et une
application web qui regroupe les objectifs des membres du groupe.

## Objectifs

| N° | Objectif | Membre | État |
|---|---|---|---|
| 1 | Segmentation des anomalies au pixel près | *à compléter* | terminé |
| 2 | *à compléter* | *à compléter* | en cours |
| 3 | *à compléter* | *à compléter* | en cours |
| 4 | *à compléter* | *à compléter* | en cours |
| 5 | *à compléter* | *à compléter* | en cours |
| 6 | *à compléter* | *à compléter* | en cours |

## Objectif 1 — Segmentation des anomalies

Pour chaque pixel d'une image de route, décider s'il appartient à une anomalie (segmentation
sémantique binaire). Cinq approches ont été comparées sur les mêmes 60 images, en validation
croisée en 5 plis, stratifiée par catégorie et groupée par source (pour éviter la fuite de
données entre images presque identiques).

| Modèle | Approche | AP | IoU |
|---|---|---|---|
| **C. DeepLabV3+ (ResNet34, ImageNet)** | segmentation supervisée, transfer learning | **0,830** | **0,659** |
| B. U-Net (ResNet34, ImageNet) | segmentation supervisée, transfer learning | 0,798 | 0,642 |
| D2. SegFormer Cityscapes + max-logit | détection hors distribution, sans entraînement | 0,423 | 0,282 |
| A. U-Net entraîné de zéro | CNN de référence | 0,237 | 0,200 |
| E1. Autoencodeur convolutif (erreur L1) | erreur de reconstruction | 0,206 | 0,119 |

AP = précision moyenne (hasard : 0,098). IoU calculée sur l'ensemble des pixels, au seuil choisi
sur la validation.

**À retenir :**

- le pré-entraînement sur ImageNet triple l'IoU (0,21 → 0,64) ;
- U-Net et DeepLabV3+ ne diffèrent pas significativement en IoU (test de Wilcoxon, p = 0,31) ;
- DeepLabV3+ est plus stable d'un pli à l'autre et généralise mieux aux anomalies jamais vues
  (pneus : AP 0,573 contre 0,124) : c'est le modèle déployé ;
- point faible commun : les petits objets (moins de 15 % détectés).

## Structure du dépôt

```
├── notebooks/
│   ├── 01_exploration_dataset.ipynb      exploration et statistiques du dataset
│   ├── 02_pretraitement.ipynb            masques, 5 plis, augmentation, Dataset PyTorch
│   ├── 03a_modeles_supervises.ipynb      U-Net de zéro, U-Net et DeepLabV3+ pré-entraînés
│   ├── 03b_cityscapes_incertitude.ipynb  SegFormer Cityscapes + scores d'incertitude
│   ├── 03c_autoencodeur.ipynb            autoencodeur convolutif
│   ├── 04_benchmark.ipynb                comparaison de tous les modèles
│   └── 05_deploiement.ipynb              entraînement final, export ONNX, application
├── src/
│   └── roadanomaly_lib.py                chargement, découpage et prétraitement partagés
├── application/                          application web Gradio (une page par objectif)
│   ├── app.py
│   ├── objectifs/objectif_1.py … objectif_6.py
│   ├── modeles/objectif_N/               modèles (non versionnés, voir plus bas)
│   ├── GUIDE_INTEGRATION.md              comment ajouter son objectif
│   └── requirements.txt
└── resultats/                            figures et tableaux produits par les notebooks
```

## Reproduire les résultats (Google Colab)

1. Télécharger le dataset RoadAnomaly et le compresser en `RoadAnomaly.zip`.
2. Ouvrir les notebooks dans l'ordre sur Colab (*Fichier → Importer un notebook*), GPU T4 activé
   à partir du notebook 03a.
3. Fichiers à envoyer dans la première cellule de chaque notebook :

| Notebook | Fichiers à envoyer | Produit |
|---|---|---|
| 01, 02 | `RoadAnomaly.zip` | `resultats_etape2.zip` (contient `roadanomaly_lib.py`) |
| 03a, 03b, 03c | `RoadAnomaly.zip`, `roadanomaly_lib.py` | `resultats_etape3*.zip` |
| 04 | les deux précédents + les trois `resultats_etape3*.zip` | `resultats_etape4.zip` |
| 05 | `RoadAnomaly.zip`, `roadanomaly_lib.py`, `resultats_etape4.zip` | modèle ONNX + application |

## Lancer l'application

```
cd application
pip install -r requirements.txt
python app.py
```

puis ouvrir http://127.0.0.1:7860. Le modèle de l'objectif 1 (`model.onnx` et `config.json`,
produits par le notebook 05) doit être placé dans `application/modeles/objectif_1/`.

**Les modèles ne sont pas dans le dépôt** (fichiers lourds) : ils sont disponibles dans les
*Releases* du dépôt, ou à régénérer avec le notebook 05.

## Données

Le dataset RoadAnomaly n'est pas inclus : ses images appartiennent à leurs auteurs et sont
fournies pour la recherche. Le télécharger depuis la page de l'EPFL et citer :

> K. Lis, K. Nakka, P. Fua, M. Salzmann. *Detecting the Unexpected via Image Resynthesis.*
> ICCV 2019.

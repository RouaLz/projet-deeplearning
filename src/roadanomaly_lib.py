"""roadanomaly_lib.py — chargement, découpage et prétraitement du dataset RoadAnomaly.
Généré par 02_pretraitement.ipynb ; réutilisé par les notebooks 03 à 05."""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

ANOMALY_LABEL = 2
CATEGORIES = ["animals", "obstacles", "cones", "lost_tires", "vehicle"]

# Images issues de la même source (cf. étape 1, §4.3) : jamais réparties entre entraînement et test
SOURCE_GROUPS = {
    "youtube_lost_tires": ["lost_tires03", "lost_tires08", "lost_tires09"],
    "cardboard_hedgehogs": ["obstacles03_cardboard_czech_hedgehog", "obstacles04_cardboard_czech_hedgehog",
                            "obstacles05_cardboard_czech_hedgehog", "obstacles06_cardboard_czech_hedgehog"],
    "tripadvisor_rocks": ["obstacles12_rocks2", "obstacles12_rocks4", "obstacles12_rocks5", "obstacles12_rocks6"],
    "papakura_news": ["cones07", "cones08_can"],
    "broadmoor": ["cones02_Broadmoor_Manhole_Cones", "vehicle02_Broadmoor_Cones_Skidloader"],
}
_GROUPE = {n: g for g, noms in SOURCE_GROUPS.items() for n in noms}

def list_frames(root):
    with open(Path(root) / "frame_list.json") as f:
        return [Path(x).stem for x in json.load(f)]

def category_of(name):
    m = re.match(r"([a-z_]+?)\d", name)
    return m.group(1) if m else name.split("_")[0]

def source_group(name):
    return _GROUPE.get(name, name)

def load_image(root, name):
    """Image RGB uint8 (720, 1280, 3)."""
    return np.array(Image.open(Path(root) / "frames" / f"{name}.webp").convert("RGB"))

def load_mask(root, name):
    """Masque binaire uint8 (720, 1280) : 1 = anomalie, 0 = fond."""
    sem = np.array(Image.open(Path(root) / "frames" / f"{name}.labels" / "labels_semantic.png"))
    return (sem == ANOMALY_LABEL).astype(np.uint8)

from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS, SEED = 5, 42

def make_splits(root, n_folds=N_FOLDS, seed=SEED):
    """Attribue chaque image à un pli (0..n_folds-1).
    Stratifié par catégorie, groupé par source : une source n'apparaît que dans un seul pli."""
    names = list_frames(root)
    df = pd.DataFrame({"name": names,
                       "category": [category_of(n) for n in names],
                       "group": [source_group(n) for n in names]})
    df["fold"] = -1
    import warnings   # avertissement attendu : la classe la plus rare (pneus) n'a que 3 images
    warnings.filterwarnings("ignore", message="The least populated class")
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(df, y=df["category"], groups=df["group"])):
        df.loc[test_idx, "fold"] = k
    return df

def fold_sets(splits, k, n_folds=N_FOLDS):
    """Tour k de la validation croisée : test = pli k, validation = pli k+1, entraînement = les 3 autres."""
    val_fold = (k + 1) % n_folds
    test  = splits.loc[splits.fold == k, "name"].tolist()
    val   = splits.loc[splits.fold == val_fold, "name"].tolist()
    train = splits.loc[~splits.fold.isin([k, val_fold]), "name"].tolist()
    return train, val, test

import albumentations as A
from albumentations.pytorch import ToTensorV2

WORK_W, WORK_H = 960, 540        # résolution de travail (75 % de 1280 x 720)
CROP = 512                       # taille des découpes d'entraînement
MEAN = (0.485, 0.456, 0.406)     # statistiques ImageNet (encodeurs pré-entraînés)
STD  = (0.229, 0.224, 0.225)

def get_train_transform(crop=CROP, p_anomaly_crop=0.5):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.8, 1.25), rotate=(-5, 5), border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.PadIfNeeded(min_height=crop, min_width=crop, border_mode=cv2.BORDER_CONSTANT),
        A.OneOf([   # moitié des découpes centrées sur une anomalie, moitié au hasard
            A.CropNonEmptyMaskIfExists(height=crop, width=crop, p=p_anomaly_crop),
            A.RandomCrop(height=crop, width=crop, p=1 - p_anomaly_crop),
        ], p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.OneOf([A.MotionBlur(blur_limit=5), A.GaussianBlur(blur_limit=(3, 5))], p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_eval_transform():
    # image entière ; complétée en bas/à droite jusqu'à un multiple de 32 (540 -> 544)
    return A.Compose([
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32,
                      position="top_left", border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def denormalize(t):
    """Tenseur (3, H, W) normalisé -> image uint8 (H, W, 3) pour l'affichage."""
    x = t.detach().cpu().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    return (x.clip(0, 1) * 255).astype(np.uint8)

import torch
from torch.utils.data import Dataset, DataLoader

class RoadAnomalyDataset(Dataset):
    """Images et masques redimensionnés une fois à WORK_W x WORK_H et gardés en mémoire.
    samples_per_image > 1 : plusieurs découpes aléatoires de chaque image par époque."""

    def __init__(self, root, names, transform, samples_per_image=1):
        self.names, self.transform, self.k = list(names), transform, samples_per_image
        self.images, self.masks = {}, {}
        for n in self.names:
            img = cv2.resize(load_image(root, n), (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(load_mask(root, n), (WORK_W, WORK_H), interpolation=cv2.INTER_NEAREST)
            self.images[n], self.masks[n] = img, msk

    def __len__(self):
        return len(self.names) * self.k

    def __getitem__(self, i):
        name = self.names[i % len(self.names)]
        out = self.transform(image=self.images[name], mask=self.masks[name])
        mask = out["mask"].float().unsqueeze(0)          # (1, H, W), valeurs 0 / 1
        return {"image": out["image"], "mask": mask, "name": name}

def make_loaders(root, splits, k, batch_size=4, samples_per_image=4, num_workers=2):
    """DataLoaders du tour k : entraînement (découpes augmentées), validation et test (images entières)."""
    train, val, test = fold_sets(splits, k)
    ds_train = RoadAnomalyDataset(root, train, get_train_transform(), samples_per_image)
    ds_val   = RoadAnomalyDataset(root, val, get_eval_transform())
    ds_test  = RoadAnomalyDataset(root, test, get_eval_transform())
    return (DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
            DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=num_workers),
            DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=num_workers))

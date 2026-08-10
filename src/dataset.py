from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEX_CATS = ["male", "female", "unknown"]
SITE_CATS = ["head/neck", "upper extremity", "lower extremity", "torso",
             "palms/soles", "oral/genital", "unknown"]


def build_transforms(size: int = 224, train: bool = True):
    """Dermoscopic images have no canonical orientation, so flips and rotation
    are label-preserving. Colour jitter is kept mild: pigmentation is diagnostic."""
    if train:
        return T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MetadataEncoder:
    """Encodes age, sex and anatomical site into a fixed-length vector.

    Missing values are represented explicitly rather than silently imputed:
    absence of a recorded site may itself be informative.
    """

    def __init__(self):
        self.age_median = None

    def fit(self, df: pd.DataFrame) -> "MetadataEncoder":
        self.age_median = float(df.age_approx.median())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.age_median is None:
            raise RuntimeError("MetadataEncoder.fit must be called first")
        age_missing = df.age_approx.isna().to_numpy(dtype=np.float32)
        age = df.age_approx.fillna(self.age_median).to_numpy(dtype=np.float32) / 100.0
        sex = df.sex.fillna("unknown")
        site = df.anatom_site_general_challenge.fillna("unknown")
        parts = [age[:, None], age_missing[:, None]]
        parts += [(sex == c).to_numpy(dtype=np.float32)[:, None] for c in SEX_CATS]
        parts += [(site == c).to_numpy(dtype=np.float32)[:, None] for c in SITE_CATS]
        return np.concatenate(parts, axis=1).astype(np.float32)

    @property
    def dim(self) -> int:
        return 2 + len(SEX_CATS) + len(SITE_CATS)


class MelanomaDataset(Dataset):
    """Returns (image, metadata_vector, target) for one lesion."""

    def __init__(self, df: pd.DataFrame, img_dir: Path, meta: np.ndarray, transform):
        self.names = df.image_name.to_numpy()
        self.targets = df.target.to_numpy(dtype=np.float32)
        self.img_dir = Path(img_dir)
        self.meta = meta
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        img = Image.open(self.img_dir / f"{self.names[i]}.jpg").convert("RGB")
        return (self.transform(img),
                torch.from_numpy(self.meta[i]),
                torch.tensor(self.targets[i]))


def make_loaders(csv: Path, split_csv: Path, img_dir: Path, fold_col: str,
                 batch_size: int = 32, size: int = 224, balanced: bool = False,
                 workers: int = 8):
    """Build train/val loaders for one split protocol.

    balanced=True enables WeightedRandomSampler, which resamples the minority
    class up to roughly even batches without discarding majority examples.
    """
    df = pd.read_csv(csv)
    folds = pd.read_csv(split_csv)[["image_name", fold_col]]
    df = df.merge(folds, on="image_name")

    tr_df = df[df[fold_col] == "train"].reset_index(drop=True)
    va_df = df[df[fold_col] == "val"].reset_index(drop=True)

    enc = MetadataEncoder().fit(tr_df)
    tr_ds = MelanomaDataset(tr_df, img_dir, enc.transform(tr_df), build_transforms(size, True))
    va_ds = MelanomaDataset(va_df, img_dir, enc.transform(va_df), build_transforms(size, False))

    if balanced:
        counts = tr_df.target.value_counts()
        w = tr_df.target.map(lambda t: 1.0 / counts[t]).to_numpy()
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(tr_df), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=workers, pin_memory=True, drop_last=True)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               num_workers=workers, pin_memory=True, drop_last=True)

    va_loader = DataLoader(va_ds, batch_size=batch_size * 2, shuffle=False,
                           num_workers=workers, pin_memory=True)
    return tr_loader, va_loader, enc.dim

"""Everything data related: reading the csv, transforms/augmentation, dataloaders.

What the competition gives us:
  train.csv -> emotion (0-6) + pixels (a string of 2304 numbers = one 48x48 grayscale face)
  test.csv  -> just pixels, no labels (this is what we predict for the submission)

I parse the pixel strings into 48x48 uint8 arrays once at the start and keep them around
in a Dataset, instead of re-parsing every time.
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from .utils import NUM_CLASSES

IMG_SIZE = 48
# mean/std of the training pixels (after scaling to [0,1]). used to normalize the images.
FER_MEAN = 0.5077
FER_STD = 0.2550


def _pixels_to_image(pixel_str: str) -> np.ndarray:
    """turn "34 12 ..." (2304 numbers) into a 48x48 uint8 image."""
    arr = np.array(pixel_str.split(), dtype=np.uint8)
    return arr.reshape(IMG_SIZE, IMG_SIZE)


class FER2013Dataset(Dataset):
    """just holds the already-parsed images (N, 48, 48) and the labels (if we have them)."""

    def __init__(self, images: np.ndarray, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # "L" = 8-bit grayscale. need a PIL image so the torchvision transforms work on it.
        pil = Image.fromarray(self.images[idx], mode="L")
        x = self.transform(pil)
        if self.labels is not None:
            return x, int(self.labels[idx])
        return x


def build_transforms(train: bool, augment: bool, img_size: int = IMG_SIZE):
    """training = with augmentation, eval = plain so it's deterministic."""
    norm = T.Normalize(mean=[FER_MEAN], std=[FER_STD])
    if train and augment:
        return T.Compose([
            T.RandomHorizontalFlip(),                       # faces are roughly symmetric, so this is safe
            T.RandomRotation(10),                           # people tilt their heads a little
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0), antialias=True),
            T.ToTensor(),
            norm,
            T.RandomErasing(p=0.25),                        # hides random patches so it can't just memorize pixels
        ])
    tfs = []
    if img_size != IMG_SIZE:
        tfs.append(T.Resize(img_size, antialias=True))
    tfs += [T.ToTensor(), norm]
    return T.Compose(tfs)


def _load_images_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    images = np.stack([_pixels_to_image(p) for p in df["pixels"]])
    labels = df["emotion"].to_numpy(dtype=np.int64) if "emotion" in df.columns else None
    return images, labels


def get_dataloaders(train_csv, batch_size=128, val_split=0.1, augment=True,
                    img_size=IMG_SIZE, num_workers=2, seed=42, subset=None):
    """90/10 stratified split of train.csv.

    Gives back (train_loader, val_loader, train_labels). subset just keeps the first N
    rows - useful for quick debugging or for deliberately overfitting a small set.
    """
    images, labels = _load_images_labels(train_csv)
    if subset is not None:
        images, labels = images[:subset], labels[:subset]

    x_tr, x_val, y_tr, y_val = train_test_split(
        images, labels, test_size=val_split, stratify=labels, random_state=seed
    )

    train_ds = FER2013Dataset(x_tr, y_tr, build_transforms(True, augment, img_size))
    val_ds = FER2013Dataset(x_val, y_val, build_transforms(False, False, img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, y_tr


def get_test_loader(test_csv, batch_size=128, img_size=IMG_SIZE, num_workers=2):
    """loader for the test set (no labels) - only used to build the submission."""
    images, _ = _load_images_labels(test_csv)
    ds = FER2013Dataset(images, None, build_transforms(False, False, img_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, len(images)


def compute_class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """inverse-frequency weights. Disgust barely has any samples so it gets bumped up."""
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """undo the normalize so the image looks normal again when I log it to wandb."""
    return (img * FER_STD + FER_MEAN).clamp(0, 1)

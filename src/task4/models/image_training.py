"""Task 4 image training helpers used by build_model_v2.ipynb.

Only the supplied Task 4 train images enter fitting and validation. The saved
checkpoint contains the complete network and the inference transform settings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = False
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("7-11k", "fiveN", "freefried", "pube", "roblock")


class AddGaussianNoise:
    def __init__(self, sigma: float = 0.035):
        self.sigma = sigma

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image + torch.randn_like(image) * self.sigma).clamp(0, 1)


def image_transform(variant: str, *, train: bool) -> transforms.Compose:
    """Keep the 2:1 content ratio and apply augmentation on train only."""
    steps = [
        transforms.Resize((112, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.Pad((0, 56, 0, 56), fill=0),
    ]
    if train and variant in {"geometry", "geometry_noise", "full_rotation"}:
        degrees = 180 if variant == "full_rotation" else 50
        steps += [
            transforms.RandomApply(
                [transforms.RandomRotation(degrees, fill=(0, 0, 0))], p=0.65
            ),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.3, fill=0),
            transforms.RandomApply(
                [transforms.RandomAffine(0, scale=(0.8, 1.18), fill=(0, 0, 0))], p=0.4
            ),
        ]
    if train and variant == "geometry_noise":
        steps.append(transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.2, 1.2))], p=0.35))
    steps.append(transforms.ToTensor())
    if train and variant == "geometry_noise":
        steps.append(transforms.RandomApply([AddGaussianNoise()], p=0.35))
    steps.append(transforms.Normalize(MEAN, STD))
    return transforms.Compose(steps)


class CompetitionImages(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray, transform):
        self.paths = paths
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        assert len(paths) == len(labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image), int(self.labels[index])


def make_model(*, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def predict(model, paths, labels, device, batch_size=32):
    if labels is None:
        labels = np.zeros(len(paths), dtype=np.int64)
    loader = DataLoader(
        CompetitionImages(paths, labels, image_transform("none", train=False)),
        batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    output = []
    with torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(device, non_blocking=True))
            output.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(output)


def fit_image_model(
    train_paths: list[Path], train_labels: np.ndarray,
    valid_paths: list[Path], valid_labels: np.ndarray,
    *, variant: str, frozen: bool, seed: int, epochs: int,
    batch_size: int = 32, patience: int = 3,
):
    """Fit on train rows, select an epoch by untouched validation macro F1."""
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(pretrained=True).to(device)
    if frozen:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    train_loader = DataLoader(
        CompetitionImages(train_paths, train_labels, image_transform(variant, train=True)),
        batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-3 if frozen else 1e-4, weight_decay=1e-4,
    )
    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    weights = torch.tensor(
        len(train_labels) / (len(CLASS_NAMES) * class_counts),
        dtype=torch.float32, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score, best_state, best_pred, stale = -1.0, None, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        if frozen:
            model.apply(lambda module: module.eval() if isinstance(module, nn.BatchNorm2d) else None)
        loss_sum = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
        pred = predict(model, valid_paths, valid_labels, device, batch_size)
        score = f1_score(valid_labels, pred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_labels), "valid_macro_F1": float(score)})
        print(f"{variant} epoch {epoch}/{epochs}: loss={history[-1]['train_loss']:.4f}, valid_macro_F1={score:.5f}", flush=True)
        if score > best_score + 1e-5:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_pred, stale = pred.copy(), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu(), best_pred, history


def fit_full_image_model(paths, labels, *, variant: str, seed: int,
                         epochs: int, batch_size: int = 32):
    """Fit the selected procedure on all labeled Train images for inference."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(pretrained=True).to(device)
    loader = DataLoader(
        CompetitionImages(paths, labels, image_transform(variant, train=True)),
        batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    weights = torch.tensor(len(labels) / (len(CLASS_NAMES) * counts),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(targets)
        value = loss_sum / len(labels)
        history.append({"epoch": epoch, "train_loss": value})
        print(f"full {variant} epoch {epoch}/{epochs}: loss={value:.4f}", flush=True)
    return model.cpu(), history


def save_checkpoint(model, path: str):
    torch.save({
        "architecture": "torchvision_resnet18",
        "class_names": CLASS_NAMES,
        "image_size": (112, 224),
        "padding": (0, 56, 0, 56),
        "mean": MEAN, "std": STD,
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path: Path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert tuple(saved["class_names"]) == CLASS_NAMES
    model = make_model(pretrained=False)
    model.load_state_dict(saved["state_dict"])
    return model

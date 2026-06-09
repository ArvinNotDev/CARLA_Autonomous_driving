# train_junction_classifier.py

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image


@dataclass
class Sample:
    path: Path
    label: int


class JunctionDataset(Dataset):
    def __init__(self, samples: List[Sample], tfm=None):
        self.samples = samples
        self.tfm = tfm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.tfm is not None:
            img = self.tfm(img)
        label = torch.tensor(s.label, dtype=torch.long)
        return img, label


def read_samples(data_dir: Path) -> List[Sample]:
    csv_path = data_dir / "labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"labels.csv not found at: {csv_path}")

    samples: List[Sample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["filename"].strip()
            label = int(row["label"])
            path = data_dir / rel
            if path.exists():
                samples.append(Sample(path=path, label=label))

    if not samples:
        raise RuntimeError("No valid samples found. Check labels.csv paths.")
    return samples


def stratified_split(samples: List[Sample], train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for s in samples:
        by_label[s.label].append(s)

    train, val, test = [], [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_transforms(image_size=224):
    train_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfm, eval_tfm


def make_model(num_classes=2):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train(args):
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = read_samples(data_dir)
    train_samples, val_samples, test_samples = stratified_split(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_tfm, eval_tfm = build_transforms(args.image_size)

    train_ds = JunctionDataset(train_samples, tfm=train_tfm)
    val_ds = JunctionDataset(val_samples, tfm=eval_tfm)
    test_ds = JunctionDataset(test_samples, tfm=eval_tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_acc = -1.0
    best_state = None

    print(f"Device: {device}")
    print(f"Samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "image_size": args.image_size,
                "class_names": {0: "not_junction", 1: "junction"},
                "best_val_acc": best_val_acc,
                "seed": args.seed,
            }

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint state.")

    torch.save(best_state, out_path)
    print(f"Saved best checkpoint to: {out_path}")
    print(f"Best val acc: {best_val_acc:.4f}")

    # Final test evaluation with best model
    model.load_state_dict(best_state["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    metrics_path = out_path.with_suffix(".json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_acc": best_val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "test_size": len(test_ds),
                "image_size": args.image_size,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to: {metrics_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="dataset")
    p.add_argument("--out", type=str, default="junction_model_resnet18.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)
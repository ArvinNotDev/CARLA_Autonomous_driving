# test_junction_classifier.py

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

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


def build_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def make_model(num_classes=2):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    tp = tn = fp = fn = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / max(total, 1)
    return {
        "accuracy": acc,
        "total": total,
        "correct": correct,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def predict_single_image(model, image_path: Path, tfm, device):
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item())
    return pred, conf, probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--checkpoint", type=str, default="junction_model_resnet18.pt")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    image_size = int(ckpt.get("image_size", 224))

    tfm = build_eval_transform(image_size=image_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class_names = ckpt.get("class_names", {0: "not_junction", 1: "junction"})

    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        pred, conf, probs = predict_single_image(model, image_path, tfm, device)
        print(f"Image: {image_path}")
        print(f"Prediction: {pred} ({class_names.get(pred, str(pred))})")
        print(f"Confidence: {conf:.4f}")
        print(f"Probabilities: not_junction={probs[0]:.4f}, junction={probs[1]:.4f}")
        return

    samples = read_samples(data_dir)
    _, _, test_samples = stratified_split(samples, seed=args.seed)

    test_ds = JunctionDataset(test_samples, tfm=tfm)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    result = evaluate(model, test_loader, device)
    print("Test evaluation")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Total: {result['total']}")
    print(f"Correct: {result['correct']}")
    print(f"TP: {result['tp']}  TN: {result['tn']}  FP: {result['fp']}  FN: {result['fn']}")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()
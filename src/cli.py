"""Command line entry point.

    python -m src.cli train   --task classification --config configs/scratch.yaml
    python -m src.cli predict --task classification --checkpoint outputs/scratch_cnn/best_model.pth --image bird.jpg
    python -m src.cli tune    --task classification --config configs/scratch.yaml
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from PIL import Image

from src.data.cub_data import make_classification_loaders, get_transforms
from src.models.bird_cnn import BirdCNN
from src.training.train import train_one_epoch, validate
from src.training.tuning import run_tuning
from src.visualization.plots import plot_history


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cub(config):
    data_cfg = config["data"]
    train_hf = load_dataset(data_cfg["train_dataset"], split="train")
    test_hf = load_dataset(data_cfg["test_dataset"], split="test")
    return train_hf, test_hf


def train(config_path, epochs_override=None):
    config = load_config(config_path)
    train_cfg = config["training"]
    epochs = epochs_override or train_cfg["epochs"]

    set_seed(train_cfg["seed"])
    device = get_device()
    torch.backends.cudnn.benchmark = True
    print("Device:", device)

    train_hf, test_hf = load_cub(config)
    data_cfg = config["data"]
    train_loader, val_loader, _ = make_classification_loaders(
        train_hf,
        test_hf,
        manifest_path=data_cfg["manifest_path"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        seed=train_cfg["seed"],
    )

    model = BirdCNN(num_classes=config["model"]["num_classes"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-2),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg["learning_rate"],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    output_dir = Path(config["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_acc = -1.0

    for epoch in range(epochs):
        start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"{time.perf_counter() - start:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": config["model"]["num_classes"],
                    "epoch": epoch + 1,
                    "val_accuracy": val_acc,
                },
                output_dir / "best_model.pth",
            )
            print("Saved new best model")

        # Save every epoch so nothing is lost if Colab disconnects.
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    plot_history(history, output_dir)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


def predict(checkpoint_path, image_path):
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = BirdCNN(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = get_transforms(train=False)(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)
        top_prob, top_class = probabilities.topk(5, dim=1)

    print("Top-5 predictions (class ID: confidence)")
    for p, c in zip(top_prob[0].tolist(), top_class[0].tolist()):
        print(f"  {c}: {p * 100:.2f}%")


def tune(config_path, epochs_override=None):
    config = load_config(config_path)
    train_hf, test_hf = load_cub(config)
    run_tuning(
        train_hf,
        test_hf,
        config["data"]["manifest_path"],
        epochs=epochs_override or config["training"]["epochs"],
    )


def main():
    parser = argparse.ArgumentParser(description="Bird Classification CNN")
    parser.add_argument("command", choices=["train", "predict", "tune"])
    parser.add_argument("--task", default="classification")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--image")
    parser.add_argument("--epochs", type=int, help="override epochs from the config")
    args = parser.parse_args()

    if args.task != "classification":
        parser.error("Only classification is implemented")

    if args.command == "train":
        if not args.config:
            parser.error("--config is required for training")
        train(args.config, args.epochs)
    elif args.command == "predict":
        if not args.checkpoint or not args.image:
            parser.error("--checkpoint and --image are required")
        predict(args.checkpoint, args.image)
    elif args.command == "tune":
        if not args.config:
            parser.error("--config is required for tuning")
        tune(args.config, args.epochs)


if __name__ == "__main__":
    main()
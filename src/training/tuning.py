
import csv
import itertools
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.cub_data import make_classification_loaders
from src.models.bird_cnn import BirdCNN


SEED = 42
NUM_CLASSES = 200
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1

OUTPUT_DIR = Path("outputs/tuning_current")
RESULTS_PATH = OUTPUT_DIR / "tuning_results.csv"

FIELDS = [
    "method",
    "trial",
    "learning_rate",
    "batch_size",
    "epochs",
    "best_epoch",
    "best_val_accuracy",
    "duration_seconds",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_trial(
    train_hf,
    test_hf,
    manifest_path,
    learning_rate,
    batch_size,
    epochs,
    device,
):
    # Give every configuration the same starting seed.
    set_seed(SEED)

    train_loader, val_loader, _ = make_classification_loaders(
        train_hf,
        test_hf,
        manifest_path=manifest_path,
        batch_size=batch_size,
        num_workers=0,
        seed=SEED,
    )

    # Fresh CNN for every experiment.
    model = BirdCNN(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    best_val_accuracy = -1.0
    best_epoch = 0

    start = time.perf_counter()

    for epoch in range(epochs):

        # ---------------------------
        # Training
        # ---------------------------
        model.train()

        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            scheduler.step()

            count = labels.size(0)

            train_loss_sum += loss.item() * count
            train_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            train_total += count

        train_loss = train_loss_sum / train_total
        train_accuracy = (
            100.0 * train_correct / train_total
        )

        # ---------------------------
        # Validation
        # ---------------------------
        model.eval()

        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                with torch.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                count = labels.size(0)

                val_loss_sum += loss.item() * count
                val_correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()
                val_total += count

        val_loss = val_loss_sum / val_total
        val_accuracy = (
            100.0 * val_correct / val_total
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%",
            flush=True,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

    duration = time.perf_counter() - start

    del model, optimizer, scheduler, scaler
    del train_loader, val_loader

    if use_amp:
        torch.cuda.empty_cache()

    return best_val_accuracy, best_epoch, duration


def save_results(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )
        writer.writeheader()
        writer.writerows(results)


def run_tuning(
    train_hf,
    test_hf,
    manifest_path,
    epochs=3,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    set_seed(SEED)

    print("Device:", device)
    print("Epochs per trial:", epochs)

    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # ---------------------------
    # Grid Search
    # ---------------------------
    grid_configs = list(
        itertools.product(
            [0.001, 0.003],
            [32, 64],
        )
    )

    # ---------------------------
    # Random Search
    # ---------------------------
    rng = random.Random(SEED)

    random_configs = []

    for _ in range(4):
        lr = math.exp(
            rng.uniform(
                math.log(0.0001),
                math.log(0.003),
            )
        )

        batch_size = rng.choice([32, 64])

        random_configs.append(
            (lr, batch_size)
        )

    experiments = [
        ("grid", grid_configs),
        ("random", random_configs),
    ]

    # Save an empty CSV with the correct header.
    save_results(results)

    # ---------------------------
    # Run all experiments
    # ---------------------------
    for method, configurations in experiments:

        print("\n" + "=" * 50)
        print(method.upper(), "SEARCH")
        print("=" * 50)

        for trial, (lr, batch_size) in enumerate(
            configurations,
            start=1,
        ):
            print(
                f"\n{method.upper()} trial {trial}: "
                f"LR={lr:.6f}, "
                f"Batch Size={batch_size}",
                flush=True,
            )

            accuracy, best_epoch, duration = run_trial(
                train_hf=train_hf,
                test_hf=test_hf,
                manifest_path=manifest_path,
                learning_rate=lr,
                batch_size=batch_size,
                epochs=epochs,
                device=device,
            )

            result = {
                "method": method,
                "trial": trial,
                "learning_rate": round(lr, 8),
                "batch_size": batch_size,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_val_accuracy": round(
                    accuracy, 4
                ),
                "duration_seconds": round(
                    duration, 2
                ),
            }

            results.append(result)

            # Save progress after every completed trial.
            save_results(results)

            print(
                f"Best Val Acc: {accuracy:.2f}% | "
                f"Duration: {duration:.1f}s",
                flush=True,
            )

    # ---------------------------
    # Compare methods
    # ---------------------------
    print("\n" + "=" * 50)
    print("FINAL TUNING COMPARISON")
    print("=" * 50)

    for method in ["grid", "random"]:
        subset = [
            row for row in results
            if row["method"] == method
        ]

        best = max(
            subset,
            key=lambda row: row["best_val_accuracy"],
        )

        total_time = sum(
            row["duration_seconds"]
            for row in subset
        )

        print(
            f"\n{method.upper()} BEST:\n"
            f"Accuracy: {best['best_val_accuracy']:.2f}%\n"
            f"Learning rate: {best['learning_rate']}\n"
            f"Batch size: {best['batch_size']}\n"
            f"Best epoch: {best['best_epoch']}\n"
            f"Trials: {len(subset)}\n"
            f"Total time: {total_time:.1f}s"
        )

    print("\nResults saved to:", RESULTS_PATH)

    return results

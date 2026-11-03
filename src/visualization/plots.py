from pathlib import Path
import matplotlib.pyplot as plt


def plot_history(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(output_dir / "loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_accuracy"], label="Train")
    plt.plot(epochs, history["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(output_dir / "accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
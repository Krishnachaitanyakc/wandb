"""Simple training script used by the sandbox sweep example.

This script is uploaded to the sandbox as part of the code artifact.
It reads hyperparameters from the active wandb run config (populated by
``wandb agent``) and logs a fake validation loss so the sweep can tune.
"""

import math

import wandb


def train() -> None:
    with wandb.init() as run:
        lr: float = run.config.get("lr", 1e-3)
        batch_size: int = run.config.get("batch_size", 32)
        epochs: int = run.config.get("epochs", 5)

        for epoch in range(1, epochs + 1):
            # Simulate a loss that decreases with a well-chosen lr
            val_loss = math.exp(-lr * 1000 * epoch) + 0.05 * (batch_size / 128)
            run.log({"epoch": epoch, "val_loss": val_loss})

        print(f"Finished: lr={lr}, batch_size={batch_size}, final val_loss={val_loss:.4f}")


if __name__ == "__main__":
    train()

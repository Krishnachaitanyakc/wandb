"""Sandbox sweep example.

Demonstrates how to run a W&B hyperparameter sweep where each trial
executes inside an isolated W&B sandbox rather than on the local machine.

Steps
-----
1. Upload the training code as a W&B artifact.
2. Create a Bayesian sweep.
3. Bring a sandbox configured to your needs (GPU, image, etc.).
4. Call ``wandb.agent_consume_sandbox`` to decorate the sandbox and start
   the agent loop inside it.

Requirements
------------
    pip install "wandb[sandbox]"

Usage
-----
    python run_sweep.py --entity <entity> --project <project>
"""

import argparse

import wandb
from wandb.sandbox import Session


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr": {"min": 1e-4, "max": 1e-1},
        "batch_size": {"values": [16, 32, 64, 128]},
        "epochs": {"value": 5},
    },
}


def upload_code_artifact(entity: str, project: str) -> str:
    """Log the training code as a versioned artifact and return its path."""
    with wandb.init(entity=entity, project=project, job_type="upload-code") as run:
        artifact = wandb.Artifact(
            name="training-code",
            type="code",
            description="Training script for the sandbox sweep example.",
        )
        artifact.add_file("train.py")
        run.log_artifact(artifact)

    return f"{entity}/{project}/training-code:latest"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sweep inside a W&B sandbox.")
    parser.add_argument("--entity", required=True, help="W&B entity (username or team).")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument(
        "--image",
        default="python:3.11-slim",
        help="Container image for the sandbox (default: python:3.11-slim).",
    )
    args = parser.parse_args()

    # 1. Upload the training code
    artifact_id = upload_code_artifact(args.entity, args.project)
    print(f"Code artifact: {artifact_id}")

    # 2. Create the sweep
    sweep_id = wandb.sweep(
        SWEEP_CONFIG,
        entity=args.entity,
        project=args.project,
    )
    print(f"Sweep created: {args.entity}/{args.project}/{sweep_id}")

    # 3. Configure the sandbox, then start it
    with Session() as session:
        sandbox = session.sandbox(container_image=args.image)

        # consume_sandbox decorates the sandbox in-place (sets tags, credentials,
        # and the startup command) without starting it.
        wandb.agent_consume_sandbox(
            sweep_id=sweep_id,
            entity=args.entity,
            project=args.project,
            artifact_id=artifact_id,
        ).consume_sandbox(sandbox)

        # Now start — the sandbox downloads the artifact and runs wandb agent.
        sandbox.start()


if __name__ == "__main__":
    main()

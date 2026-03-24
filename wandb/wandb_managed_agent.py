"""Managed sweep agent that configures a W&B sandbox to run a sweep.

``ManagedAgent.consume_sandbox`` decorates a sandbox *in-place* — setting
tags, credentials, and a startup command — without starting it.  The caller
then starts the sandbox however they like and it automatically downloads the
artifact code and runs ``wandb agent``.

Usage::

    import wandb
    from wandb.sandbox import Session

    # 1. Log training code as an artifact
    with wandb.init(project="mnist", entity="acme") as run:
        code = wandb.Artifact("training-code", type="code")
        code.add_dir(".")
        run.log_artifact(code)

    # 2. Create a sweep
    sweep_id = wandb.sweep(sweep_cfg, project="mnist", entity="acme")

    # 3. Configure the sandbox, then start it
    with Session() as session:
        sandbox = session.sandbox(container_image="pytorch/pytorch:latest")

        wandb.agent_consume_sandbox(
            sweep_id=sweep_id,
            project="mnist",
            entity="acme",
            artifact_id="acme/mnist/training-code:latest",
        ).consume_sandbox(sandbox)

        sandbox.start()   # sandbox downloads code and runs wandb agent
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import wandb
from wandb.apis import InternalApi

if TYPE_CHECKING:
    from wandb.sandbox._sandbox import Sandbox

logger = logging.getLogger(__name__)

_ARTIFACT_SANDBOX_ROOT = "/app"


class ManagedAgent:
    """Configures a W&B sandbox to download artifact code and run a sweep agent.

    Create via :func:`agent_consume_sandbox`, then call
    :meth:`consume_sandbox` to decorate a sandbox before starting it.
    """

    def __init__(
        self,
        sweep_id: str,
        entity: str,
        project: str,
        artifact_id: str,
    ) -> None:
        self.sweep_id = sweep_id
        self.entity = entity
        self.project = project
        self.artifact_id = artifact_id
        self._api = InternalApi()

    def consume_sandbox(self, sandbox: Sandbox) -> Sandbox:
        """Decorate *sandbox* in-place so that starting it runs the sweep agent.

        Modifies the sandbox before it is started:

        * Adds a ``wandb-sweep:<sweep_id>`` tag for discoverability.
        * Injects W&B credentials (``WANDB_API_KEY``, ``WANDB_PROJECT``,
          ``WANDB_ENTITY``, ``WANDB_BASE_URL``) as environment variables.
        * Replaces the sandbox command with a startup script that downloads
          the artifact via ``wandb artifact get`` and then runs
          ``wandb agent``.

        The sandbox must not have been started yet.

        Args:
            sandbox: An unstarted :class:`wandb.sandbox.Sandbox` (or the
                base ``cwsandbox.Sandbox``) to configure.

        Returns:
            The same *sandbox* instance (for method chaining).
        """
        self._apply_tag(sandbox)
        self._apply_env_vars(sandbox)
        self._apply_startup_command(sandbox)
        return sandbox

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_tag(self, sandbox: Sandbox) -> None:
        sweep_tag = f"wandb-sweep:{self.sweep_id}"
        if sandbox._tags is None:
            sandbox._tags = [sweep_tag]
        elif sweep_tag not in sandbox._tags:
            sandbox._tags = list(sandbox._tags) + [sweep_tag]

    def _apply_env_vars(self, sandbox: Sandbox) -> None:
        updates: dict[str, str] = {
            "WANDB_PROJECT": self.project,
            "WANDB_ENTITY": self.entity,
        }
        api_key = self._api.api_key
        if api_key:
            updates["WANDB_API_KEY"] = api_key
        base_url = self._api.settings("base_url")
        if base_url:
            updates["WANDB_BASE_URL"] = base_url

        existing = dict(sandbox._environment_variables or {})
        existing.update(updates)
        sandbox._environment_variables = existing

    def _apply_startup_command(self, sandbox: Sandbox) -> None:
        sweep_path = f"{self.entity}/{self.project}/{self.sweep_id}"
        startup = (
            f"pip install --quiet wandb && "
            f"wandb artifact get {self.artifact_id} -d {_ARTIFACT_SANDBOX_ROOT} && "
            f"cd {_ARTIFACT_SANDBOX_ROOT} && "
            f"wandb agent {sweep_path}"
        )
        sandbox._command = "/bin/sh"
        sandbox._args = ["-c", startup]

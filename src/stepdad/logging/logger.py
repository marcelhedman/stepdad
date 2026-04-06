"""
Pluggable logging backends.

Usage
-----
    from stepdad.logging.logger import make_logger

    logger = make_logger("stdout")   # or "wandb"
    logger.log({"eig_lower": 7.04, "step": 1000}, step=1000)
    logger.finish()

Both backends accept the same dict-based interface so experiment scripts need
only change the --logger argument.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Logger(ABC):
    """Abstract logger interface."""

    @abstractmethod
    def log(self, metrics: dict, step: int) -> None:
        """Log a dict of scalar metrics at the given step."""
        ...

    def finish(self) -> None:
        """Called once at the end of a run.  No-op by default."""
        pass


class StdoutLogger(Logger):
    """Prints metrics to stdout."""

    def log(self, metrics: dict, step: int) -> None:
        parts = [f"step={step}"] + [f"{k}={v:.4g}" for k, v in metrics.items()]
        print("  ".join(parts))


class WandbLogger(Logger):
    """Logs metrics to Weights & Biases.

    Args:
        project: W&B project name.
        name:    run name (optional).
        config:  hyperparameter dict logged to the run summary.
    """

    def __init__(
        self,
        project: str = "stepdad",
        name: str | None = None,
        config: dict | None = None,
    ) -> None:
        import wandb
        self._run = wandb.init(project=project, name=name, config=config or {})

    def log(self, metrics: dict, step: int) -> None:
        import wandb
        # Do not pass step as the W&B x-axis — it is non-monotonic across
        # the train_dad / run_stepdad boundary.  Instead log it as a plain
        # metric so it is still queryable, and let W&B auto-increment.
        wandb.log({**metrics, "step_tracking": step})

    def finish(self) -> None:
        import wandb
        wandb.finish()


def make_logger(
    backend: str,
    *,
    project: str = "stepdad",
    name: str | None = None,
    config: dict | None = None,
) -> Logger:
    """Factory for loggers.

    Args:
        backend: "stdout" or "wandb".
        project: W&B project name (only used when backend="wandb").
        name:    run name (only used when backend="wandb").
        config:  hyperparameter dict (only used when backend="wandb").
    Returns:
        Logger instance.
    """
    if backend == "stdout":
        return StdoutLogger()
    elif backend == "wandb":
        return WandbLogger(project=project, name=name, config=config)
    else:
        raise ValueError(f"Unknown logger backend: {backend!r}. Choose 'stdout' or 'wandb'.")

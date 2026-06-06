from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from reentry_mpc.artifacts import plt
from reentry_mpc.phase8 import TARGET_NAMES


@dataclass(frozen=True)
class ResidualModelConfig:
    hidden_sizes: list[int]
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int


@dataclass(frozen=True)
class Phase9Config:
    seed: int
    dataset_dir: Path
    output_dir: Path
    model: ResidualModelConfig


class MLPResidualModel(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int = 3, hidden_sizes: list[int] | None = None
    ) -> None:
        super().__init__()
        widths = hidden_sizes or [64, 64]
        layers: list[nn.Module] = []
        previous = input_dim
        for width in widths:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.Tanh())
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def load_phase9_config(path: str | Path) -> Phase9Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    model = raw["model"]
    return Phase9Config(
        seed=int(raw["seed"]),
        dataset_dir=Path(raw["dataset_dir"]),
        output_dir=Path(raw["output_dir"]),
        model=ResidualModelConfig(
            hidden_sizes=[int(value) for value in model["hidden_sizes"]],
            learning_rate=float(model["learning_rate"]),
            weight_decay=float(model["weight_decay"]),
            batch_size=int(model["batch_size"]),
            epochs=int(model["epochs"]),
            patience=int(model["patience"]),
        ),
    )


def run_phase9_residual_model(
    config_path: str | Path = "configs/phase9_residual_model.yaml",
) -> dict[str, Path | dict[str, Any]]:
    config = load_phase9_config(config_path)
    set_deterministic_seed(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_npz_split(config.dataset_dir / "train.npz")
    val = load_npz_split(config.dataset_dir / "val.npz")
    test = load_npz_split(config.dataset_dir / "test.npz")
    normalizer = fit_normalizer(train["X"], train["y"])
    model = MLPResidualModel(
        input_dim=train["X"].shape[1],
        output_dim=train["y"].shape[1],
        hidden_sizes=config.model.hidden_sizes,
    )
    history = train_model(model, train, val, normalizer, config)
    test_predictions = predict(model, test["X"], normalizer)
    train_predictions = predict(model, train["X"], normalizer)
    val_predictions = predict(model, val["X"], normalizer)

    learned_metrics = evaluate_predictions(test["y"], test_predictions, "mlp_residual")
    zero_predictions = np.zeros_like(test["y"])
    zero_metrics = evaluate_predictions(test["y"], zero_predictions, "zero_residual")
    metrics = {
        "seed": config.seed,
        "target_names": TARGET_NAMES,
        "train": evaluate_predictions(train["y"], train_predictions, "mlp_residual"),
        "val": evaluate_predictions(val["y"], val_predictions, "mlp_residual"),
        "test": learned_metrics,
        "zero_residual_baseline": zero_metrics,
        "improves_over_zero_mse": bool(learned_metrics["mse"] < zero_metrics["mse"]),
        "mse_improvement_fraction": float(
            1.0 - learned_metrics["mse"] / max(zero_metrics["mse"], 1.0e-12)
        ),
        "normalizer": normalizer_to_json(normalizer),
        "history": history,
        "binned_error_by_mach": binned_mae(
            values=test["mach"], errors=test_predictions - test["y"]
        ),
        "binned_error_by_altitude": binned_mae(
            values=test["altitude_m"], errors=test_predictions - test["y"]
        ),
    }
    metrics_path = output_dir / "residual_model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    checkpoint_path = output_dir / "residual_model_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": train["X"].shape[1],
                "output_dim": train["y"].shape[1],
                "hidden_sizes": config.model.hidden_sizes,
            },
            "normalizer": normalizer,
            "target_names": TARGET_NAMES,
        },
        checkpoint_path,
    )
    summary_path = output_dir / "residual_model_summary.md"
    summary_path.write_text(write_summary(metrics), encoding="utf-8")
    figure_paths = write_phase9_figures(
        history=history,
        test=test,
        predictions=test_predictions,
        output_dir=output_dir,
        metrics=metrics,
    )
    return {
        "metrics_json": metrics_path,
        "summary_md": summary_path,
        "checkpoint": checkpoint_path,
        "metrics": metrics,
        **figure_paths,
    }


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_npz_split(path: Path) -> dict[str, np.ndarray]:
    split = np.load(path, allow_pickle=False)
    return {
        "X": split["X"].astype(np.float32),
        "y": split["y"].astype(np.float32),
        "mach": split["mach"].astype(np.float32),
        "altitude_m": split["altitude_m"].astype(np.float32),
        "delta_flap_rad": split["delta_flap_rad"].astype(np.float32),
    }


def fit_normalizer(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "x_mean": X.mean(axis=0),
        "x_std": np.maximum(X.std(axis=0), 1.0e-8),
        "y_mean": y.mean(axis=0),
        "y_std": np.maximum(y.std(axis=0), 1.0e-8),
    }


def normalize_X(X: np.ndarray, normalizer: dict[str, np.ndarray]) -> np.ndarray:
    return (X - normalizer["x_mean"]) / normalizer["x_std"]


def normalize_y(y: np.ndarray, normalizer: dict[str, np.ndarray]) -> np.ndarray:
    return (y - normalizer["y_mean"]) / normalizer["y_std"]


def denormalize_y(y_norm: np.ndarray, normalizer: dict[str, np.ndarray]) -> np.ndarray:
    return y_norm * normalizer["y_std"] + normalizer["y_mean"]


def make_loader(
    split: dict[str, np.ndarray],
    normalizer: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X = torch.tensor(normalize_X(split["X"], normalizer), dtype=torch.float32)
    y = torch.tensor(normalize_y(split["y"], normalizer), dtype=torch.float32)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: MLPResidualModel,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    normalizer: dict[str, np.ndarray],
    config: Phase9Config,
) -> dict[str, list[float]]:
    train_loader = make_loader(train, normalizer, config.model.batch_size, shuffle=True)
    val_loader = make_loader(val, normalizer, config.model.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )
    loss_fn = nn.MSELoss()
    best_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    best_val = float("inf")
    stale_epochs = 0
    history = {"train_loss": [], "val_loss": []}
    for _epoch in range(config.model.epochs):
        model.train()
        train_losses: list[float] = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_normalized_loss(model, val_loader, loss_fn)
        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if stale_epochs >= config.model.patience:
            break
    model.load_state_dict(best_state)
    return history


def evaluate_normalized_loss(
    model: MLPResidualModel, loader: DataLoader, loss_fn: nn.Module
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            losses.append(float(loss_fn(model(X_batch), y_batch).detach().cpu()))
    return float(np.mean(losses))


def predict(
    model: MLPResidualModel, X: np.ndarray, normalizer: dict[str, np.ndarray]
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_norm = torch.tensor(normalize_X(X, normalizer), dtype=torch.float32)
        y_norm = model(X_norm).detach().cpu().numpy()
    return denormalize_y(y_norm, normalizer).astype(np.float32)


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> dict[str, Any]:
    error = y_pred - y_true
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    nrmse = float(np.sqrt(mse) / max(float(np.std(y_true)), 1.0e-12))
    per_state = {}
    for idx, target in enumerate(TARGET_NAMES):
        target_error = error[:, idx]
        per_state[target] = {
            "mse": float(np.mean(target_error**2)),
            "mae": float(np.mean(np.abs(target_error))),
            "nrmse": float(
                np.sqrt(np.mean(target_error**2))
                / max(float(np.std(y_true[:, idx])), 1.0e-12)
            ),
        }
    return {
        "model": model_name,
        "mse": mse,
        "mae": mae,
        "normalized_rmse": nrmse,
        "per_state": per_state,
    }


def binned_mae(
    *, values: np.ndarray, errors: np.ndarray, bins: int = 6
) -> list[dict[str, float | int]]:
    edges = np.linspace(float(values.min()), float(values.max()), bins + 1)
    rows: list[dict[str, float | int]] = []
    for idx in range(bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == bins - 1:
            mask = (values >= low) & (values <= high)
        else:
            mask = (values >= low) & (values < high)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": int(mask.sum()),
                "mae": float(np.mean(np.abs(errors[mask]))),
            }
        )
    return rows


def normalizer_to_json(normalizer: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: value.astype(float).tolist() for key, value in normalizer.items()}


def write_summary(metrics: dict[str, Any]) -> str:
    test = metrics["test"]
    zero = metrics["zero_residual_baseline"]
    improved = metrics["improves_over_zero_mse"]
    mlp_row = (
        f"| MLP residual | {test['mse']:.6e} | {test['mae']:.6e} | "
        f"{test['normalized_rmse']:.4f} |"
    )
    zero_row = (
        f"| Zero residual | {zero['mse']:.6e} | {zero['mae']:.6e} | "
        f"{zero['normalized_rmse']:.4f} |"
    )
    return "\n".join(
        [
            "# Residual Model Summary",
            "",
            "The Phase 9 MLP residual model predicts `x_dot_true - x_dot_nominal`",
            "from scheduled state/control/aerodynamic features.",
            "",
            "| Model | Test MSE | Test MAE | Normalized RMSE |",
            "|---|---:|---:|---:|",
            mlp_row,
            zero_row,
            "",
            f"Improves over zero residual by MSE: `{improved}`.",
            "",
            f"MSE improvement fraction: `{metrics['mse_improvement_fraction']:.4f}`.",
            "",
            "Interpretation: a positive improvement means the network learned",
            "repeatable structure in the sampled model mismatch. It does not mean",
            "the residual model is safe to close the loop around without further",
            "rollout testing.",
            "",
        ]
    )


def write_phase9_figures(
    *,
    history: dict[str, list[float]],
    test: dict[str, np.ndarray],
    predictions: np.ndarray,
    output_dir: Path,
    metrics: dict[str, Any],
) -> dict[str, Path]:
    loss_path = output_dir / "train_val_loss.png"
    pred_path = output_dir / "predicted_vs_true_residuals.png"
    mach_path = output_dir / "residual_error_by_mach.png"
    altitude_path = output_dir / "residual_error_by_altitude.png"

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MSE")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for idx, target in enumerate(TARGET_NAMES):
        axes[idx].scatter(test["y"][:, idx], predictions[:, idx], s=12, alpha=0.6)
        lo = min(float(test["y"][:, idx].min()), float(predictions[:, idx].min()))
        hi = max(float(test["y"][:, idx].max()), float(predictions[:, idx].max()))
        axes[idx].plot([lo, hi], [lo, hi], color="black", lw=1)
        axes[idx].set_title(target)
        axes[idx].set_xlabel("True")
        axes[idx].set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(pred_path, dpi=160)
    plt.close(fig)

    write_binned_plot(metrics["binned_error_by_mach"], mach_path, "Mach")
    write_binned_plot(
        metrics["binned_error_by_altitude"], altitude_path, "Altitude (m)"
    )
    return {
        "train_val_loss_figure": loss_path,
        "predicted_vs_true_figure": pred_path,
        "residual_error_by_mach_figure": mach_path,
        "residual_error_by_altitude_figure": altitude_path,
    }


def write_binned_plot(
    rows: list[dict[str, float | int]], path: Path, xlabel: str
) -> None:
    centers = [(float(row["bin_low"]) + float(row["bin_high"])) * 0.5 for row in rows]
    values = [float(row["mae"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(centers, values, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean absolute residual error")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

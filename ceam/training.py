"""CEAM training, model selection, and held-out evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import Split
from .model import CEAM


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    hidden_units: int = 6
    learning_rate: float = 2e-4
    epochs: int = 50
    batch_size: int = 16
    correction_penalty: float = 1e-8
    penalty_form: str = "squared"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_model(train: Split, validation: Split, config: TrainingConfig) -> tuple[CEAM, dict]:
    """Optimize validation-selected CEAM parameters.

    By default, apply lambda times the BATCH-SUMMED squared L2 norm of the
    neural correction; the cross-entropy term is batch-averaged. Thus the
    effective relative penalty changes with batch size. This is the
    objective implemented by the reference training routine.
    An optional unsquared-norm penalty is also supported.
    """
    if config.penalty_form not in {"norm", "squared"}:
        raise ValueError("penalty_form must be 'norm' or 'squared'")
    set_seed(config.seed)
    model = CEAM(config.hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0)
    loader = DataLoader(TensorDataset(train.features, train.labels, train.availability),
                        batch_size=config.batch_size, shuffle=True)
    history: list[dict] = []
    best_loss = float("inf")
    best_weights: dict | None = None
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for features, target, availability in loader:
            optimizer.zero_grad(set_to_none=True)
            c = model.components(features)
            residual = torch.linalg.vector_norm(c.neural_correction)
            penalty = residual if config.penalty_form == "norm" else residual.square()
            logits = model.mask_logits(c.logits, availability)
            loss = F.cross_entropy(logits, target) + config.correction_penalty * penalty
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(target)
        metrics = evaluate(model, validation)
        history.append({"epoch": epoch + 1, "train_objective": running_loss / len(train.labels),
                        "validation_log_loss": metrics["log_loss"], "validation_accuracy": metrics["accuracy"]})
        if metrics["log_loss"] < best_loss:
            best_loss = metrics["log_loss"]
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, {"configuration": asdict(config), "best_validation_log_loss": best_loss,
                   "history": history}


def evaluate(model: CEAM, split: Split) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        log_probs = F.log_softmax(model(split.features, split.availability), dim=1)
        chosen = log_probs.gather(1, split.labels[:, None]).squeeze(1)
        accuracy = (log_probs.argmax(dim=1) == split.labels).float().mean()
    return {"n": int(len(split.labels)), "log_likelihood": float(chosen.sum()),
            "log_loss": float(-chosen.mean()), "accuracy": float(accuracy)}

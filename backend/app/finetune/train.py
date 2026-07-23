"""Fine-tune a sentence-transformer with a contrastive objective (one config)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    name: str
    loss: str = "online_contrastive"  # contrastive | online_contrastive
    epochs: int = 2
    batch_size: int = 16
    lr: float = 2e-5
    max_pairs: int = 1600
    margin: float = 0.5


def build_loss(model, cfg: TrainConfig):
    from sentence_transformers import losses
    from sentence_transformers.losses import SiameseDistanceMetric

    # Train under cosine distance to match the cosine relevance metric we evaluate.
    cosine = SiameseDistanceMetric.COSINE_DISTANCE
    if cfg.loss == "online_contrastive":
        return losses.OnlineContrastiveLoss(model, distance_metric=cosine, margin=cfg.margin)
    return losses.ContrastiveLoss(model, distance_metric=cosine, margin=cfg.margin)


def finetune(base_model_name: str, pairs: dict, cfg: TrainConfig, output_dir: str):
    """Train and save a model; return (model, loss_history)."""
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )

    model = SentenceTransformer(base_model_name)
    train_ds = Dataset.from_dict(pairs)
    loss = build_loss(model, cfg)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no",
        report_to=[],  # we log to MLflow ourselves for full control
        disable_tqdm=True,
        seed=42,
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, loss=loss
    )
    trainer.train()

    # extract real loss curve
    loss_history = [
        {"step": e["step"], "loss": e["loss"], "epoch": e.get("epoch")}
        for e in trainer.state.log_history
        if "loss" in e
    ]
    logger.info("Fine-tune %s complete: %d logged loss points, final loss=%.4f",
                cfg.name, len(loss_history), loss_history[-1]["loss"] if loss_history else float("nan"))
    return model, loss_history

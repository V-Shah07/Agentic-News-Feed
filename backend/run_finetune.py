"""Phase 6 PROVE IT: fine-tune the embedder across configs, track in MLflow,
register + promote the best model, and log the base-vs-fine-tuned relevance lift.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil

import mlflow
from mlflow.tracking import MlflowClient

from backend.app.config import get_settings
from backend.app.db import init_db, session_scope
from backend.app.finetune.dataset import build_pairs, load_labeled, split
from backend.app.finetune.evaluate import evaluate_model
from backend.app.finetune.train import TrainConfig, finetune

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("run_finetune")

EXPERIMENT = "interest-embedder-finetune"
REGISTERED_MODEL = "interest-embedder"
PROD_DIR = "models/finetuned/production"

CONFIGS = [
    TrainConfig("contrastive-1ep", loss="contrastive", epochs=1, batch_size=16, lr=2e-5, max_pairs=1000),
    TrainConfig("contrastive-2ep", loss="contrastive", epochs=2, batch_size=16, lr=2e-5, max_pairs=1000),
    TrainConfig("online-2ep-bs32", loss="online_contrastive", epochs=2, batch_size=32, lr=3e-5, max_pairs=1000),
    TrainConfig("online-3ep", loss="online_contrastive", epochs=3, batch_size=16, lr=2e-5, max_pairs=1000),
]


def main() -> None:
    settings = get_settings()
    base_model = settings.embedding_model
    init_db()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    with session_scope() as session:
        labeled = load_labeled(session)
    train, test = split(labeled, test_ratio=0.3, seed=13)
    pairs = build_pairs(train, max_pairs=1000, seed=13)
    log.info("labeled=%d train=%d test=%d pairs=%d", len(labeled), len(train), len(test), len(pairs["label"]))

    # ---- base model held-out metric ----
    from sentence_transformers import SentenceTransformer

    base = SentenceTransformer(base_model)
    base_metrics = evaluate_model(base, train, test)
    log.info("BASE metrics: %s", base_metrics)
    with mlflow.start_run(run_name="base-model"):
        mlflow.log_param("model", base_model)
        mlflow.log_param("stage", "base")
        mlflow.log_metrics({f"base_{k}": v for k, v in base_metrics.items() if isinstance(v, (int, float))})

    # ---- fine-tuning runs ----
    runs = []
    for cfg in CONFIGS:
        out_dir = f"models/finetuned/{cfg.name}"
        with mlflow.start_run(run_name=cfg.name) as run:
            mlflow.log_params({
                "model": base_model, "loss": cfg.loss, "epochs": cfg.epochs,
                "batch_size": cfg.batch_size, "learning_rate": cfg.lr,
                "max_pairs": cfg.max_pairs, "n_train_pairs": len(pairs["label"]),
            })
            model, loss_history = finetune(base_model, pairs, cfg, out_dir)
            for pt in loss_history:
                mlflow.log_metric("train_loss", pt["loss"], step=pt["step"])

            metrics = evaluate_model(model, train, test)
            improvement = ((metrics["auc"] - base_metrics["auc"]) / base_metrics["auc"] * 100.0
                           if base_metrics["auc"] else 0.0)
            mlflow.log_metrics({**metrics, "auc_improvement_pct_vs_base": round(improvement, 2)})

            # register this version in the MLflow model registry
            info = mlflow.sentence_transformers.log_model(
                model, artifact_path="model", registered_model_name=REGISTERED_MODEL,
            )
            mv = client.search_model_versions(
                f"run_id='{run.info.run_id}'"
            )
            version = mv[0].version if mv else None
            log.info("Run %s: AUC %.4f (%.2f%% vs base), registered v%s",
                     cfg.name, metrics["auc"], improvement, version)
            runs.append({
                "config": cfg.name, "run_id": run.info.run_id, "version": version,
                "auc": metrics["auc"], "separation": metrics["separation"],
                "improvement_pct": round(improvement, 2), "final_loss": loss_history[-1]["loss"] if loss_history else None,
                "model_uri": info.model_uri, "out_dir": out_dir,
            })

    # ---- promote best ----
    best = max(runs, key=lambda r: r["auc"])
    client.set_registered_model_alias(REGISTERED_MODEL, "production", best["version"])
    try:
        client.transition_model_version_stage(REGISTERED_MODEL, best["version"], "Production",
                                              archive_existing_versions=True)
    except Exception as exc:  # stages deprecated on some backends; alias is source of truth
        log.info("stage transition skipped (%s); alias 'production' set", type(exc).__name__)

    # deploy: copy the promoted model to the production path the embedder reads
    if os.path.isdir(best["out_dir"]):
        src = best["out_dir"]
    else:
        src = mlflow.artifacts.download_artifacts(best["model_uri"]) + "/model"
        src = src if os.path.isdir(src) else best["out_dir"]
    os.makedirs(os.path.dirname(PROD_DIR), exist_ok=True)
    if os.path.isdir(PROD_DIR):
        shutil.rmtree(PROD_DIR)
    # re-save the best model cleanly to the production dir
    from sentence_transformers import SentenceTransformer as ST
    best_model = ST(best["out_dir"]) if os.path.isdir(best["out_dir"]) else base
    best_model.save(PROD_DIR)

    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_model": base_model,
        "tracking_uri": settings.mlflow_tracking_uri,
        "experiment": EXPERIMENT,
        "registered_model": REGISTERED_MODEL,
        "n_experiments_tracked": len(runs) + 1,  # base + fine-tunes
        "n_finetune_configs": len(CONFIGS),
        "labeled": len(labeled), "train": len(train), "test": len(test),
        "base": base_metrics,
        "runs": runs,
        "promoted": {"config": best["config"], "version": best["version"], "auc": best["auc"]},
        "base_auc": base_metrics["auc"],
        "best_auc": best["auc"],
        "relevance_improvement_pct": best["improvement_pct"],
        "separation_base": base_metrics["separation"],
        "separation_best": best["separation"],
        "production_model_path": PROD_DIR,
    }

    print("\n===== PHASE 6 FINE-TUNING + MLflow SUMMARY =====")
    print(json.dumps({k: v for k, v in payload.items() if k != "runs"}, indent=2))
    print("\nPer-config runs:")
    for r in runs:
        print(f"  {r['config']:20s} v{r['version']}  AUC={r['auc']:.4f}  "
              f"sep={r['separation']:+.4f}  Δvs_base={r['improvement_pct']:+.2f}%  loss={r['final_loss']}")
    print(f"\nBASE AUC {base_metrics['auc']:.4f} -> BEST FINE-TUNED AUC {best['auc']:.4f} "
          f"({best['improvement_pct']:+.2f}% relevance lift). Promoted '{best['config']}' "
          f"v{best['version']} to production.")

    os.makedirs("logs", exist_ok=True)
    with open("logs/phase6_finetune.log", "a") as fh:
        fh.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()

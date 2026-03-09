#!/usr/bin/env python3
"""Fase 1 — Baselines tabulares: RF, XGBoost, MLP, MLP→RF."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.utils import load_config, set_seed, ensure_dirs, get_logger, save_results
from src.data import (
    load_dataset, binarize_target, get_feature_columns,
    split_standard, check_leakage,
)
from src.features import fit_preprocessor, transform_tabular
from src.baselines import train_rf, train_xgboost, train_mlp, train_mlp_rf, predict_mlp_rf
from src.metrics import compute_all_metrics, measure_latency, measure_memory, aggregate_seeds


def save_history(history: list, model_name: str, seed: int):
    """Salva curvas de treino (loss, accuracy por epoch) em CSV."""
    path = Path(f"outputs/results/history_{model_name}_seed{seed}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def run_single_seed(cfg: dict, seed: int, logger):
    set_seed(seed)
    logger.info(f"=== Seed {seed} ===")

    # Data
    df = load_dataset(cfg)
    df = binarize_target(df, cfg)
    feature_cols = get_feature_columns(df, cfg)
    splits = split_standard(df, cfg, seed)
    check_leakage(splits, feature_cols)

    # Preprocessing
    artifacts = fit_preprocessor(splits["train"], feature_cols, cfg)
    X_train = transform_tabular(splits["train"], feature_cols, artifacts)
    X_val = transform_tabular(splits["val"], feature_cols, artifacts)
    X_test = transform_tabular(splits["test"], feature_cols, artifacts)
    y_train = splits["train"]["label"].values
    y_val = splits["val"]["label"].values
    y_test = splits["test"]["label"].values

    logger.info(f"Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    logger.info(f"Features={X_train.shape[1]}")

    all_results = {}

    # --- RF ---
    logger.info("Training RF...")
    rf = train_rf(X_train, y_train, cfg["baselines"]["random_forest"])
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = rf.predict(X_test)
    all_results["rf"] = compute_all_metrics(y_test, rf_preds, rf_probs)
    all_results["rf"]["model"] = "RF"
    logger.info(f"RF: PR-AUC={all_results['rf']['pr_auc']:.4f}, "
                f"F1={all_results['rf']['f1_macro']:.4f}")

    # --- XGBoost ---
    logger.info("Training XGBoost...")
    xgb = train_xgboost(X_train, y_train, cfg["baselines"]["xgboost"])
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_preds = xgb.predict(X_test)
    all_results["xgboost"] = compute_all_metrics(y_test, xgb_preds, xgb_probs)
    all_results["xgboost"]["model"] = "XGBoost"
    logger.info(f"XGBoost: PR-AUC={all_results['xgboost']['pr_auc']:.4f}, "
                f"F1={all_results['xgboost']['f1_macro']:.4f}")

    # --- MLP ---
    logger.info("Training MLP...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp, mlp_history = train_mlp(
        X_train, y_train, X_val, y_val, cfg["baselines"]["mlp"], device
    )
    save_history(mlp_history, "mlp", seed)
    logger.info(f"MLP: {len(mlp_history)} epochs, "
                f"best val_acc={max(h['val_accuracy'] for h in mlp_history):.4f}")

    mlp.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = mlp(X_t).cpu().numpy()
    
    # Stable softmax to avoid overflow/NaN
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits_shifted)
    mlp_probs = exps[:, 1] / exps.sum(axis=1)
    
    mlp_preds = logits.argmax(axis=1)
    all_results["mlp"] = compute_all_metrics(y_test, mlp_preds, mlp_probs)
    all_results["mlp"]["model"] = "MLP"
    logger.info(f"MLP: PR-AUC={all_results['mlp']['pr_auc']:.4f}, "
                f"F1={all_results['mlp']['f1_macro']:.4f}")

    # --- MLP→RF (paper base) ---
    logger.info("Training MLP→RF...")
    mlp_rf_model, rf_head, mlp_rf_history = train_mlp_rf(
        X_train, y_train, X_val, y_val, cfg["baselines"]["mlp_rf"], device
    )
    save_history(mlp_rf_history, "mlp_rf", seed)
    logger.info(f"MLP→RF: {len(mlp_rf_history)} epochs, "
                f"best val_acc={max(h['val_accuracy'] for h in mlp_rf_history):.4f}")

    mlp_rf_preds, mlp_rf_probs = predict_mlp_rf(mlp_rf_model, rf_head, X_test, device)
    mlp_rf_probs_pos = mlp_rf_probs[:, 1]
    all_results["mlp_rf"] = compute_all_metrics(y_test, mlp_rf_preds, mlp_rf_probs_pos)
    all_results["mlp_rf"]["model"] = "MLP→RF"
    logger.info(f"MLP→RF: PR-AUC={all_results['mlp_rf']['pr_auc']:.4f}, "
                f"F1={all_results['mlp_rf']['f1_macro']:.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Fase 1: Baselines tabulares")
    parser.add_argument("--config", default="/home/tec/Projects/slm-spyware-detection/configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("baselines")

    seeds = cfg["seeds"]
    results_by_model = {}

    for seed in seeds:
        seed_results = run_single_seed(cfg, seed, logger)
        for model_key, metrics in seed_results.items():
            results_by_model.setdefault(model_key, []).append(metrics)

    # Agregação
    logger.info("\n=== Resultados agregados (mean±std) ===")
    for model_key, results_list in results_by_model.items():
        agg = aggregate_seeds(results_list)
        logger.info(f"\n{model_key}:\n{agg['formatted'].to_string()}")
        save_results(results_list, f"outputs/results/baselines_{model_key}.csv")

    logger.info("Fase 1 completa.")


if __name__ == "__main__":
    main()

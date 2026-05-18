#!/usr/bin/env python3
"""Fase 3 — Método A: SLM benign-only (anomalia/zero-day)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    DEFAULT_CONFIG, load_config, set_seed, ensure_dirs, get_logger,
    resolve_device, save_results,
)
from src.data import (
    load_dataset, binarize_target, get_feature_columns,
    split_standard, split_unseen_family,
)
from src.features import serialize_dataframe
from src.slm import (
    cleanup_cuda, load_slm, finetune_benign_only, score_anomaly,
    compute_thresholds, peak_vram_mb,
)
from src.metrics import compute_anomaly_metrics


def flatten_anomaly_results(results: dict, model_name: str, seed: int, family: str = None):
    """Converte métricas por threshold em linhas tabulares para avaliação."""
    rows = []
    for threshold_key, vals in results.items():
        row = {
            "model": model_name,
            "seed": seed,
            "threshold_target": threshold_key,
            **vals,
        }
        if family is not None:
            row["family"] = family
        rows.append(row)
    return rows


def run_standard_split(cfg, seed, logger):
    """Benign-only no split padrão."""
    set_seed(seed)
    logger.info(f"=== Standard split | Seed {seed} ===")

    df = load_dataset(cfg)
    df = binarize_target(df, cfg)
    feature_cols = get_feature_columns(df, cfg)
    splits = split_standard(df, cfg, seed)

    # Serializa
    benign_train = splits["train"][splits["train"]["label"] == 0]
    benign_val = splits["val"][splits["val"]["label"] == 0]
    texts_benign_train = serialize_dataframe(benign_train, feature_cols, cfg)
    texts_benign_val = serialize_dataframe(benign_val, feature_cols, cfg)
    texts_test = serialize_dataframe(splits["test"], feature_cols, cfg)
    y_test = splits["test"]["label"].values

    logger.info(f"Loading SLM: {cfg['slm']['active_model']} (fresh model for seed)")
    model, tokenizer = load_slm(cfg, quantize=True, causal_lm=True)

    # Fine-tune nos benignos
    logger.info(f"Fine-tuning on {len(texts_benign_train)} benign samples...")
    model = finetune_benign_only(model, tokenizer, texts_benign_train, cfg,
                                  texts_val=texts_benign_val)

    # Score
    logger.info("Scoring test set...")
    device = resolve_device(cfg["slm"].get("device", "cuda"))
    batch_size = cfg["slm"]["batch_size"]
    scores_val = score_anomaly(model, tokenizer, texts_benign_val, cfg,
                                batch_size, device)
    cleanup_cuda()
    scores_test = score_anomaly(model, tokenizer, texts_test, cfg,
                                 batch_size, device)
    logger.info(f"Peak VRAM benign-only: {peak_vram_mb():.0f}MB")

    # Thresholds via validação benigna
    fpr_targets = cfg["slm"]["benign_only"]["thresholds"]["fpr_targets"]
    thresholds = compute_thresholds(scores_val, fpr_targets)
    logger.info(f"Thresholds: {thresholds}")

    # Métricas
    results = compute_anomaly_metrics(y_test, scores_test, thresholds)
    for key, vals in results.items():
        logger.info(f"  {key}: TPR={vals['tpr']:.4f}, F1={vals['f1']:.4f}")

    del model, tokenizer
    cleanup_cuda()
    return results


def run_unseen_family(cfg, seed, logger):
    """Benign-only com protocolo unseen-family."""
    set_seed(seed)
    logger.info(f"=== Unseen-family | Seed {seed} ===")

    df = load_dataset(cfg)
    df = binarize_target(df, cfg)
    feature_cols = get_feature_columns(df, cfg)

    folds = split_unseen_family(df, cfg, seed)
    all_fold_results = []

    for fold in folds:
        family = fold["family"]
        logger.info(f"--- Held-out family: {family} ---")

        benign_train = fold["train"][fold["train"]["label"] == 0]
        benign_val = fold["val"][fold["val"]["label"] == 0]
        texts_benign_train = serialize_dataframe(benign_train, feature_cols, cfg)
        texts_benign_val = serialize_dataframe(benign_val, feature_cols, cfg)
        texts_test = serialize_dataframe(fold["test"], feature_cols, cfg)
        y_test = fold["test"]["label"].values

        logger.info(f"Benign train={len(texts_benign_train)}, Test={len(y_test)} "
                     f"(malware={y_test.sum()})")

        logger.info(f"Loading SLM: {cfg['slm']['active_model']} (fresh model for fold)")
        model, tokenizer = load_slm(cfg, quantize=True, causal_lm=True)

        # Fine-tune
        model_ft = finetune_benign_only(model, tokenizer, texts_benign_train, cfg,
                                         texts_val=texts_benign_val)

        # Score
        device = resolve_device(cfg["slm"].get("device", "cuda"))
        batch_size = cfg["slm"]["batch_size"]
        scores_val = score_anomaly(model_ft, tokenizer, texts_benign_val, cfg,
                                    batch_size, device)
        cleanup_cuda()
        scores_test = score_anomaly(model_ft, tokenizer, texts_test, cfg,
                                     batch_size, device)
        logger.info(f"Peak VRAM fold: {peak_vram_mb():.0f}MB")

        fpr_targets = cfg["slm"]["benign_only"]["thresholds"]["fpr_targets"]
        thresholds = compute_thresholds(scores_val, fpr_targets)

        results = compute_anomaly_metrics(y_test, scores_test, thresholds)
        all_fold_results.extend(
            flatten_anomaly_results(results, "SLM benign-only unseen", seed, family)
        )

        for key, vals in results.items():
            logger.info(f"  {key}: TPR={vals['tpr']:.4f}")

        del model, model_ft, tokenizer
        cleanup_cuda()

    return all_fold_results


def main():
    parser = argparse.ArgumentParser(description="Fase 3: SLM benign-only")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--profile", choices=["16gb"], help="Aplicar profile de VRAM")
    parser.add_argument("--unseen-family", action="store_true",
                        help="Rodar protocolo unseen-family")
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile)
    ensure_dirs(cfg)
    logger = get_logger("slm_benign")

    seeds = cfg["seeds"]


    if args.unseen_family:
        all_results = []
        for seed in seeds:
            all_results.extend(run_unseen_family(cfg, seed, logger))
        save_results(all_results, f"{cfg['paths']['results']}/benign_only_unseen.csv")
    else:
        all_results = []
        for seed in seeds:
            results = run_standard_split(cfg, seed, logger)
            all_results.extend(
                flatten_anomaly_results(results, "SLM benign-only standard", seed)
            )
        save_results(all_results, f"{cfg['paths']['results']}/benign_only_standard.csv")

    logger.info("Fase 3 completa.")


if __name__ == "__main__":
    main()

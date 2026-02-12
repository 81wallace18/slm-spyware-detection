#!/usr/bin/env python3
"""Fase 2 — Método B: SLM embeddings + head clássico."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.utils import load_config, set_seed, ensure_dirs, get_logger, save_results
from src.data import (
    load_dataset, binarize_target, get_feature_columns,
    split_standard, check_leakage,
)
from src.features import serialize_dataframe, check_token_lengths
from src.slm import load_slm, extract_embeddings, train_embedding_head
from src.metrics import (
    compute_all_metrics, measure_latency, measure_memory, aggregate_seeds,
)


def run_single_seed(cfg: dict, seed: int, model, tokenizer, logger, quantized: bool = False):
    set_seed(seed)
    tag = "4bit" if quantized else "fp16"
    logger.info(f"=== Seed {seed} ({tag}) ===")

    # Data
    df = load_dataset(cfg)
    df = binarize_target(df, cfg)
    feature_cols = get_feature_columns(df, cfg)
    splits = split_standard(df, cfg, seed)
    check_leakage(splits, feature_cols)

    # Serialização
    texts_train = serialize_dataframe(splits["train"], feature_cols, cfg)
    texts_val = serialize_dataframe(splits["val"], feature_cols, cfg)
    texts_test = serialize_dataframe(splits["test"], feature_cols, cfg)

    # Checa tokens (só na primeira seed)
    if seed == cfg["seeds"][0]:
        stats = check_token_lengths(texts_train[:500], tokenizer,
                                     cfg["slm"]["max_seq_len"])
        logger.info(f"Token stats (train sample): {stats}")

    # Embeddings
    logger.info("Extracting embeddings...")
    batch_size = cfg["slm"]["batch_size"]
    device = cfg["slm"]["device"]

    emb_train = extract_embeddings(model, tokenizer, texts_train, cfg, batch_size, device)
    emb_val = extract_embeddings(model, tokenizer, texts_val, cfg, batch_size, device)
    emb_test = extract_embeddings(model, tokenizer, texts_test, cfg, batch_size, device)
    logger.info(f"Embedding dim: {emb_train.shape[1]}")

    y_train = splits["train"]["label"].values
    y_test = splits["test"]["label"].values

    # Treina heads
    all_results = {}
    emb_cfg = cfg["slm"]["embedding"]

    for head_name in emb_cfg["heads"]:
        logger.info(f"Training head: {head_name}")
        params = emb_cfg["head_params"].get(head_name, {})
        head = train_embedding_head(emb_train, y_train, head_name, params)

        probs = head.predict_proba(emb_test)[:, 1]
        preds = head.predict(emb_test)

        key = f"slm_embed_{head_name}_{tag}"
        all_results[key] = compute_all_metrics(y_test, preds, probs)
        all_results[key]["model"] = f"SLM+{head_name} ({tag})"
        logger.info(f"{head_name}: PR-AUC={all_results[key]['pr_auc']:.4f}, "
                     f"F1={all_results[key]['f1_macro']:.4f}")

    # Memory
    mem = measure_memory()
    logger.info(f"Memory: RAM={mem['ram_mb']:.0f}MB, VRAM={mem['vram_mb']:.0f}MB")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Fase 2: SLM embeddings + head")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--quantize", action="store_true", help="Usar modelo 4-bit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("slm_embed")

    # Carrega modelo uma vez
    logger.info(f"Loading SLM: {cfg['slm']['active_model']} "
                f"(quantized={args.quantize})")
    model, tokenizer = load_slm(cfg, quantize=args.quantize)

    seeds = cfg["seeds"]
    results_by_model = {}

    for seed in seeds:
        seed_results = run_single_seed(cfg, seed, model, tokenizer, logger,
                                        quantized=args.quantize)
        for model_key, metrics in seed_results.items():
            results_by_model.setdefault(model_key, []).append(metrics)

    # Agregação
    logger.info("\n=== Resultados agregados ===")
    for model_key, results_list in results_by_model.items():
        agg = aggregate_seeds(results_list)
        logger.info(f"\n{model_key}:\n{agg['formatted'].to_string()}")
        save_results(results_list, f"outputs/results/{model_key}.csv")

    logger.info("Fase 2 completa.")


if __name__ == "__main__":
    main()

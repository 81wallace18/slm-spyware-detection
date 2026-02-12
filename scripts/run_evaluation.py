#!/usr/bin/env python3
"""Fase 4 — Avaliação final: tabelas comparativas, plots, métricas de deploy."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from src.utils import load_config, ensure_dirs, get_logger


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Carrega todos os CSVs de resultados."""
    results_path = Path(results_dir)
    dfs = []
    for f in results_path.glob("baselines_*.csv"):
        df = pd.read_csv(f)
        dfs.append(df)
    for f in results_path.glob("slm_embed_*.csv"):
        df = pd.read_csv(f)
        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def generate_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Gera tabela mean±std agrupada por modelo."""
    if df.empty:
        return df

    metrics_cols = ["pr_auc", "roc_auc", "f1_macro", "precision", "recall",
                     "fpr_at_tpr_95"]
    available = [c for c in metrics_cols if c in df.columns]

    grouped = df.groupby("model")[available].agg(["mean", "std"])

    # Formata como mean±std
    rows = []
    for model in grouped.index:
        row = {"model": model}
        for metric in available:
            m = grouped.loc[model, (metric, "mean")]
            s = grouped.loc[model, (metric, "std")]
            row[metric] = f"{m:.4f}±{s:.4f}"
        rows.append(row)

    return pd.DataFrame(rows)


def generate_delta_table(comparison: pd.DataFrame, baseline_model: str = "MLP→RF"):
    """Calcula delta % relativo ao paper base."""
    if comparison.empty or baseline_model not in comparison["model"].values:
        return comparison

    baseline_row = comparison[comparison["model"] == baseline_model].iloc[0]
    metrics_cols = [c for c in comparison.columns if c != "model"]

    rows = []
    for _, row in comparison.iterrows():
        delta_row = {"model": row["model"]}
        for metric in metrics_cols:
            base_val = float(baseline_row[metric].split("±")[0])
            curr_val = float(row[metric].split("±")[0])
            if base_val > 0:
                delta_pct = ((curr_val - base_val) / base_val) * 100
                delta_row[f"Δ_{metric}_%"] = f"{delta_pct:+.2f}%"
        rows.append(delta_row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Fase 4: Avaliação final")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("evaluation")

    results_dir = cfg["paths"]["results"]

    # Tabela comparativa
    df = load_all_results(results_dir)
    if df.empty:
        logger.warning("Nenhum resultado encontrado. Rode as fases 1-3 primeiro.")
        return

    comparison = generate_comparison_table(df)
    logger.info("\n=== Tabela comparativa ===")
    logger.info(f"\n{comparison.to_string(index=False)}")

    comparison.to_csv(f"{results_dir}/comparison_table.csv", index=False)

    # Delta vs paper base
    delta = generate_delta_table(comparison)
    logger.info("\n=== Delta vs MLP→RF (paper base) ===")
    logger.info(f"\n{delta.to_string(index=False)}")

    delta.to_csv(f"{results_dir}/delta_table.csv", index=False)

    logger.info(f"\nTabelas salvas em {results_dir}/")
    logger.info("Fase 4 completa.")


if __name__ == "__main__":
    main()

"""Métricas de classificação e de deploy."""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from typing import Dict, Callable, Optional
import psutil


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def fpr_at_tpr(y_true: np.ndarray, y_prob: np.ndarray, tpr_target: float = 0.95) -> float:
    """Calcula FPR quando TPR >= tpr_target."""
    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    # Encontra o menor FPR onde TPR >= target
    valid = tpr_arr >= tpr_target
    if valid.any():
        return float(fpr_arr[valid][0])
    return 1.0


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Calcula todas as métricas de classificação."""
    # y_prob = probabilidade da classe positiva
    results = {
        "pr_auc": pr_auc(y_true, y_prob),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_class_0": float(f1_score(y_true, y_pred, average="binary", pos_label=0)),
        "f1_class_1": float(f1_score(y_true, y_pred, average="binary", pos_label=1)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "fpr_at_tpr_95": fpr_at_tpr(y_true, y_prob, 0.95),
    }

    cm = confusion_matrix(y_true, y_pred)
    results["tn"] = int(cm[0, 0])
    results["fp"] = int(cm[0, 1])
    results["fn"] = int(cm[1, 0])
    results["tp"] = int(cm[1, 1])

    return results


def compute_anomaly_metrics(
    y_true: np.ndarray, scores: np.ndarray,
    thresholds: Dict[float, float]
) -> Dict[str, Dict[str, float]]:
    """Métricas para o método benign-only (score-based)."""
    results = {}
    for fpr_target, threshold in thresholds.items():
        y_pred = (scores >= threshold).astype(int)
        results[f"fpr_{fpr_target}"] = {
            "threshold": threshold,
            "tpr": float(recall_score(y_true, y_pred)),
            "fpr_actual": float(1 - precision_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred)),
            "pr_auc": pr_auc(y_true, scores),
        }
    return results


# ── Métricas de deploy ───────────────────────────────────────

def measure_latency(
    predict_fn: Callable, X_sample: np.ndarray, n_runs: int = 100
) -> Dict[str, float]:
    """Mede latência e throughput."""
    # Warmup
    for _ in range(5):
        predict_fn(X_sample[:1])

    # Latência por amostra
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(X_sample[:1])
        times.append(time.perf_counter() - start)

    # Throughput (batch completo)
    start = time.perf_counter()
    predict_fn(X_sample)
    batch_time = time.perf_counter() - start

    return {
        "latency_mean_ms": float(np.mean(times) * 1000),
        "latency_p95_ms": float(np.percentile(times, 95) * 1000),
        "throughput_samples_per_sec": float(len(X_sample) / batch_time),
    }


def measure_memory() -> Dict[str, float]:
    """Mede uso de RAM e VRAM."""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024 / 1024

    vram_mb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    except ImportError:
        pass

    return {"ram_mb": float(ram_mb), "vram_mb": float(vram_mb)}


def aggregate_seeds(results_per_seed: list) -> pd.DataFrame:
    """Agrega resultados de múltiplas seeds em mean±std."""
    df = pd.DataFrame(results_per_seed)
    agg = pd.DataFrame({
        "mean": df.mean(),
        "std": df.std(),
    })
    agg["formatted"] = agg.apply(
        lambda r: f"{r['mean']:.4f}±{r['std']:.4f}", axis=1
    )
    return agg

"""Preprocessing tabular e serialização texto para SLM."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Tuple


# ── Tabular ──────────────────────────────────────────────────

def fit_preprocessor(
    df_train: pd.DataFrame, feature_cols: List[str], cfg: dict
) -> dict:
    """Ajusta scaler no treino, retorna artefatos."""
    strategy = cfg["preprocessing"]["numeric"]["strategy"]
    scalers = {
        "zscore": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    scaler = scalers[strategy]()

    numeric_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    scaler.fit(df_train[numeric_cols])

    return {"scaler": scaler, "numeric_cols": numeric_cols}


def transform_tabular(
    df: pd.DataFrame, feature_cols: List[str], artifacts: dict
) -> np.ndarray:
    """Aplica scaler e retorna array NumPy."""
    df_out = df[feature_cols].copy()
    nc = artifacts["numeric_cols"]
    df_out[nc] = artifacts["scaler"].transform(df_out[nc])
    return df_out.values.astype(np.float32)


# ── Serialização texto ───────────────────────────────────────

def serialize_row(row: pd.Series, feature_cols: List[str], cfg: dict) -> str:
    """Converte uma amostra em string 'key=value ; key=value ; ...'."""
    ser_cfg = cfg["serialization"]
    precision = ser_cfg["numeric_precision"]
    sep = ser_cfg["separator"]

    sorted_cols = sorted(feature_cols) if ser_cfg["sort_keys"] else feature_cols
    parts = []
    for col in sorted_cols:
        val = row[col]
        if isinstance(val, (int, float, np.integer, np.floating)):
            parts.append(f"{col}={val:.{precision}f}")
        else:
            parts.append(f"{col}={val}")
    return sep.join(parts)


def serialize_dataframe(
    df: pd.DataFrame, feature_cols: List[str], cfg: dict
) -> List[str]:
    """Serializa todo o DataFrame em lista de strings."""
    return [serialize_row(row, feature_cols, cfg) for _, row in df.iterrows()]


def check_token_lengths(
    texts: List[str], tokenizer, max_tokens: int
) -> dict:
    """Verifica distribuição de tokens e quantos excedem max_tokens."""
    lengths = [len(tokenizer.encode(t)) for t in texts]
    lengths = np.array(lengths)
    exceeded = (lengths > max_tokens).sum()
    return {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "max": int(lengths.max()),
        "min": int(lengths.min()),
        "p95": float(np.percentile(lengths, 95)),
        "exceeded_max": int(exceeded),
        "exceeded_pct": float(exceeded / len(lengths) * 100),
    }

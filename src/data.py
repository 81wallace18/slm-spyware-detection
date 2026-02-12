"""Data loading, cleaning e splitting para CIC-MalMem-2022."""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from typing import Tuple, Dict, List, Optional


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Carrega CSV e faz limpeza básica."""
    ds_cfg = cfg["dataset"]
    df = pd.read_csv(ds_cfg["path"])

    # Remove duplicatas
    if cfg["preprocessing"].get("remove_duplicates", False):
        df = df.drop_duplicates()

    # Remove features constantes
    if cfg["preprocessing"].get("remove_constant_features", False):
        nunique = df.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        df = df.drop(columns=constant_cols, errors="ignore")

    return df


def binarize_target(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Converte target para 0 (benign) / 1 (malware)."""
    ds_cfg = cfg["dataset"]
    target = ds_cfg["target_column"]

    if ds_cfg["task"] == "binary":
        df["label"] = (df[target] == ds_cfg["positive_label"]).astype(int)
    else:
        df["label"] = df[target]

    return df


def get_feature_columns(df: pd.DataFrame, cfg: dict) -> List[str]:
    """Retorna lista de features (exclui target, drop_columns, label)."""
    ds_cfg = cfg["dataset"]
    exclude = set(ds_cfg.get("drop_columns", []))
    exclude.add(ds_cfg["target_column"])
    exclude.add("label")
    return [c for c in df.columns if c not in exclude]


def split_standard(
    df: pd.DataFrame, cfg: dict, seed: int
) -> Dict[str, pd.DataFrame]:
    """Split estratificado train/val/test."""
    s_cfg = cfg["splits"]["standard"]
    labels = df["label"]

    # Primeiro: separa test
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=s_cfg["test"], random_state=seed
    )
    trainval_idx, test_idx = next(sss1.split(df, labels))

    # Segundo: separa val do trainval
    val_ratio = s_cfg["val"] / (s_cfg["train"] + s_cfg["val"])
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=seed
    )
    df_trainval = df.iloc[trainval_idx]
    train_idx, val_idx = next(sss2.split(df_trainval, df_trainval["label"]))

    return {
        "train": df_trainval.iloc[train_idx].reset_index(drop=True),
        "val": df_trainval.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def split_unseen_family(
    df: pd.DataFrame, cfg: dict, seed: int
) -> List[Dict[str, pd.DataFrame]]:
    """Leave-one-family-out: treina sem a família, testa nela.

    Retorna lista de dicts, um por família held-out.
    Cada dict tem train/val/test onde test contém APENAS a família removida
    + amostras benignas do test padrão.
    """
    uf_cfg = cfg["splits"]["unseen_family"]
    family_col = uf_cfg["family_column"]

    if family_col not in df.columns:
        raise ValueError(f"Coluna '{family_col}' não encontrada no dataset")

    # Famílias de malware (exclui benigno)
    malware_mask = df["label"] == 1
    families = df.loc[malware_mask, family_col].unique().tolist()

    folds = []
    for held_family in families:
        held_mask = (df[family_col] == held_family) & malware_mask
        rest = df[~held_mask]

        # Split padrão no rest
        s_cfg = cfg["splits"]["standard"]
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=s_cfg["test"], random_state=seed
        )
        trainval_idx, test_base_idx = next(sss.split(rest, rest["label"]))

        val_ratio = s_cfg["val"] / (s_cfg["train"] + s_cfg["val"])
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=seed
        )
        rest_trainval = rest.iloc[trainval_idx]
        train_idx, val_idx = next(
            sss2.split(rest_trainval, rest_trainval["label"])
        )

        # Test = amostras benignas do test padrão + toda a família held-out
        test_benign = rest.iloc[test_base_idx]
        test_benign = test_benign[test_benign["label"] == 0]
        test_held = df[held_mask]
        test = pd.concat([test_benign, test_held], ignore_index=True)

        folds.append({
            "family": held_family,
            "train": rest_trainval.iloc[train_idx].reset_index(drop=True),
            "val": rest_trainval.iloc[val_idx].reset_index(drop=True),
            "test": test.reset_index(drop=True),
        })

    return folds


def check_leakage(splits: Dict[str, pd.DataFrame], feature_cols: List[str]):
    """Verifica se há vazamento entre train e test."""
    train_idx = set(splits["train"].index)
    test_idx = set(splits["test"].index)
    overlap = train_idx & test_idx
    if overlap:
        raise ValueError(f"Leakage detectado: {len(overlap)} amostras em train E test")
    print(f"[OK] Sem leakage. Train={len(splits['train'])}, "
          f"Val={len(splits['val'])}, Test={len(splits['test'])}")

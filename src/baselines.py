"""Baselines tabulares: RF, XGBoost, MLP, MLP→RF."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple, Optional


# ── Random Forest ────────────────────────────────────────────

def train_rf(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    rf = RandomForestClassifier(**params, random_state=42)
    rf.fit(X_train, y_train)
    return rf


# ── XGBoost ──────────────────────────────────────────────────

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    from xgboost import XGBClassifier

    p = {k: v for k, v in params.items() if k != "n_jobs"}
    clf = XGBClassifier(**p, n_jobs=params.get("n_jobs", -1),
                         use_label_encoder=False, eval_metric="logloss",
                         random_state=42)
    clf.fit(X_train, y_train)
    return clf


# ── MLP ──────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list,
                 dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def extract_embedding(self, x):
        return self.backbone(x)


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict, device: str = "cuda"
) -> Tuple[MLP, list]:
    """Retorna (modelo, history) onde history é lista de dicts por epoch."""
    input_dim = X_train.shape[1]
    model = MLP(
        input_dim=input_dim,
        hidden_layers=params["hidden_layers"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_dl = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=params["batch_size"])

    best_val_loss = float("inf")
    patience_counter = 0
    es = params.get("early_stopping", {})
    patience = es.get("patience", 5)
    history = []

    for epoch in range(params["epochs"]):
        # Train
        model.train()
        train_losses = []
        correct, total = 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        train_loss = np.mean(train_losses)
        train_acc = correct / total

        # Val
        model.eval()
        val_losses = []
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += len(yb)
        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


# ── MLP → RF (paper base) ───────────────────────────────────

def train_mlp_rf(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict, device: str = "cuda"
) -> Tuple[MLP, RandomForestClassifier, list]:
    """Treina MLP, extrai embeddings da penúltima camada, treina RF.
    Retorna (mlp, rf, history)."""
    mlp, history = train_mlp(X_train, y_train, X_val, y_val, params["mlp"], device)

    # Extrai embeddings
    mlp.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        embeddings = mlp.extract_embedding(X_t).cpu().numpy()

    # Treina RF nos embeddings
    rf = RandomForestClassifier(**params["rf"], random_state=42)
    rf.fit(embeddings, y_train)

    return mlp, rf, history


def predict_mlp_rf(
    mlp: MLP, rf: RandomForestClassifier,
    X: np.ndarray, device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """Predição do pipeline MLP→RF."""
    mlp.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        embeddings = mlp.extract_embedding(X_t).cpu().numpy()
    preds = rf.predict(embeddings)
    probs = rf.predict_proba(embeddings)
    return preds, probs

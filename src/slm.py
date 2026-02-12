"""SLM: embeddings + head clássico (método B) e benign-only scoring (método A)."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional, Dict
from pathlib import Path


# ── Dataset de texto ─────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_len: int,
                 labels: Optional[np.ndarray] = None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Carregar modelo SLM ─────────────────────────────────────

def load_slm(cfg: dict, quantize: bool = False):
    """Carrega modelo e tokenizer. Se quantize=True, carrega em 4-bit."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["slm"]["active_model"]
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    if quantize and cfg["slm"]["quantization"]["enabled"]:
        from transformers import BitsAndBytesConfig
        qcfg = cfg["slm"]["quantization"]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


# ── Método B: Embeddings ─────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model, tokenizer, texts: List[str], cfg: dict,
    batch_size: int = 16, device: str = "cuda"
) -> np.ndarray:
    """Extrai embeddings via mean pooling do last hidden state."""
    max_len = cfg["slm"]["max_seq_len"]
    ds = TextDataset(texts, tokenizer, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)

        # Mean pooling com mask
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        embedding = (summed / counts).cpu().numpy()
        all_embeddings.append(embedding)

    return np.concatenate(all_embeddings, axis=0)


def train_embedding_head(
    X_train: np.ndarray, y_train: np.ndarray, head_name: str, params: dict
):
    """Treina head clássico sobre embeddings do SLM."""
    if head_name == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(**params, use_label_encoder=False,
                             eval_metric="logloss", random_state=42)
    elif head_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(**params, random_state=42)
    elif head_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(**params, random_state=42)
    else:
        raise ValueError(f"Head desconhecido: {head_name}")

    clf.fit(X_train, y_train)
    return clf


# ── Método A: Benign-only scoring ────────────────────────────

def finetune_benign_only(
    model, tokenizer, texts_benign: List[str], cfg: dict,
    texts_val: Optional[List[str]] = None,
):
    """Fine-tune causal LM com LoRA nos textos benignos."""
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    bo_cfg = cfg["slm"]["benign_only"]
    lora_cfg = bo_cfg["lora"]

    if lora_cfg["enabled"]:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    max_len = cfg["slm"]["max_seq_len"]
    train_ds = TextDataset(texts_benign, tokenizer, max_len)
    val_ds = TextDataset(texts_val, tokenizer, max_len) if texts_val else None

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="outputs/models/benign_only",
        num_train_epochs=bo_cfg["epochs"],
        per_device_train_batch_size=cfg["slm"]["batch_size"],
        learning_rate=bo_cfg["learning_rate"],
        logging_steps=50,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        load_best_model_at_end=bool(val_ds),
        report_to="none",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    return model


@torch.no_grad()
def score_anomaly(
    model, tokenizer, texts: List[str], cfg: dict,
    batch_size: int = 16, device: str = "cuda"
) -> np.ndarray:
    """Calcula mean token NLL por amostra (score de anomalia)."""
    max_len = cfg["slm"]["max_seq_len"]
    ds = TextDataset(texts, tokenizer, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    scores = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # (B, seq_len-1, vocab)
        targets = input_ids[:, 1:]           # (B, seq_len-1)

        # Loss por token
        B, S, V = logits.shape
        token_loss = loss_fn(logits.reshape(B * S, V), targets.reshape(B * S))
        token_loss = token_loss.reshape(B, S)

        # Mask: só tokens reais (não padding)
        mask = attention_mask[:, 1:].float()
        mean_nll = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        scores.append(mean_nll.cpu().numpy())

    return np.concatenate(scores)


def compute_thresholds(
    scores_benign: np.ndarray, fpr_targets: List[float]
) -> Dict[float, float]:
    """Calcula thresholds para FPR-alvo usando scores de validação benigna."""
    thresholds = {}
    for fpr in fpr_targets:
        # Percentil alto = threshold (benignos devem ter score baixo)
        percentile = (1 - fpr) * 100
        thresholds[fpr] = float(np.percentile(scores_benign, percentile))
    return thresholds

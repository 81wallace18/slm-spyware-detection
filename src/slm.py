"""SLM: embeddings + head clássico (método B) e benign-only scoring (método A)."""

import os
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Dict

# Otimização de fragmentação de VRAM para GPUs pequenas
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


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
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Carregar modelo SLM ─────────────────────────────────────

def load_slm(cfg: dict, quantize: bool = False, causal_lm: bool = True):
    """Carrega modelo e tokenizer. Se quantize=True, carrega em 4-bit.
    causal_lm=False carrega apenas o modelo base (útil para embeddings)."""
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

    model_name = cfg["slm"]["active_model"]
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    if quantize and cfg["slm"]["quantization"]["enabled"]:
        from transformers import BitsAndBytesConfig
        q_cfg = cfg["slm"]["quantization"]
        compute_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(q_cfg.get("compute_dtype", "float16"), torch.float16)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q_cfg.get("quant_type", "fp4"),
            bnb_4bit_use_double_quant=q_cfg.get("use_double_quant", False),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Se causal_lm=True, carrega com o head de linguagem (para Método A/Fine-tuning)
    # Se causal_lm=False, carrega apenas o encoder base (para Método B/Embeddings)
    loader = AutoModelForCausalLM if causal_lm else AutoModel
    model = loader.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def cleanup_cuda():
    """Libera cache de CUDA sem falhar em CPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def peak_vram_mb(reset: bool = True) -> float:
    if not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return float(peak)


def forward_no_cache(model, **kwargs):
    try:
        return model(**kwargs, use_cache=False)
    except TypeError:
        return model(**kwargs)


# ── Método B: Embeddings ─────────────────────────────────────

@torch.inference_mode()
def extract_embeddings(
    model, tokenizer, texts: List[str], cfg: dict,
    batch_size: int = 8, device: str = "cuda"
) -> np.ndarray:
    """Extrai embeddings via mean pooling do last hidden state."""
    from transformers import DataCollatorWithPadding

    max_len = cfg["slm"]["max_seq_len"]
    ds = TextDataset(texts, tokenizer, max_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    all_embeddings = []
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Usar o modelo diretamente (sem output_hidden_states para economizar VRAM)
        outputs = forward_no_cache(
            model, input_ids=input_ids, attention_mask=attention_mask
        )
        
        # Identifica onde está o last_hidden_state dependendo de como o modelo foi carregado
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif hasattr(model, "model") and hasattr(model.model, "forward"):
            # Se for CausalLM, acessamos o modelo base interno para pegar o hidden state
            # sem precisar da camada de predição final (LM Head)
            hidden = forward_no_cache(
                model.model, input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        else:
            raise ValueError("O modelo fornecido não permite extração direta de hidden states.")

        # Mean pooling com mask
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        embedding = (summed / counts).cpu().numpy()
        all_embeddings.append(embedding)
        del input_ids, attention_mask, outputs, hidden, mask_expanded, summed, counts
        cleanup_cuda()

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
    """Fine-tune causal LM com LoRA nos textos benignos (QLoRA)."""
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    bo_cfg = cfg["slm"]["benign_only"]
    lora_cfg = bo_cfg["lora"]

    # Preparação para treinamento quantizado (crucial para 4-bit)
    if cfg["slm"]["quantization"]["enabled"]:
        model = prepare_model_for_kbit_training(model)

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

    # Ativar checkpointing de gradientes para economizar VRAM (troca tempo por memória)
    model.gradient_checkpointing_enable()

    max_len = cfg["slm"]["max_seq_len"]
    train_ds = TextDataset(texts_benign, tokenizer, max_len)
    val_ds = TextDataset(texts_val, tokenizer, max_len) if texts_val else None

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_strategy = bo_cfg.get("eval_strategy", "epoch" if val_ds else "no")
    save_strategy = bo_cfg.get("save_strategy", "epoch")
    load_best = bo_cfg.get("load_best_model_at_end", bool(val_ds))

    # Configurações de treinamento otimizadas para VRAM baixa
    training_args = TrainingArguments(
        output_dir=f"{cfg['paths']['models']}/benign_only",
        num_train_epochs=bo_cfg["epochs"],
        per_device_train_batch_size=bo_cfg.get(
            "per_device_train_batch_size", cfg["slm"]["batch_size"]
        ),
        per_device_eval_batch_size=bo_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=bo_cfg.get("gradient_accumulation_steps", 16),
        eval_accumulation_steps=1,     # Limpa VRAM frequentemente na avaliação
        prediction_loss_only=True,     # Não guarda logits (economiza MUITA VRAM)
        optim="paged_adamw_8bit",
        learning_rate=bo_cfg["learning_rate"],
        logging_steps=10,
        eval_strategy=eval_strategy if val_ds else "no",
        save_strategy=save_strategy,
        load_best_model_at_end=load_best and val_ds is not None,
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    
    # Desativa cache durante o treino (economiza VRAM e evita erros com checkpointing)
    model.config.use_cache = False
    
    trainer.train()
    
    # Reativa cache após o treino para inferência rápida
    model.config.use_cache = True
    
    return model


@torch.inference_mode()
def score_anomaly(
    model, tokenizer, texts: List[str], cfg: dict,
    batch_size: int = 16, device: str = "cuda"
) -> np.ndarray:
    """Calcula mean token NLL por amostra (score de anomalia)."""
    from transformers import DataCollatorWithPadding

    max_len = cfg["slm"]["max_seq_len"]
    ds = TextDataset(texts, tokenizer, max_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    scores = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = forward_no_cache(
            model, input_ids=input_ids, attention_mask=attention_mask
        )
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
        del input_ids, attention_mask, outputs, logits, targets, token_loss, mask, mean_nll
        cleanup_cuda()

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

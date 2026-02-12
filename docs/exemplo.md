# Blueprint — Hugo: SLM-first Spyware Detection

> Todas as decisões estão fechadas. Este documento é a referência final para implementação.

---

## Decisões fechadas

| Item | Decisão |
|---|---|
| Dataset | **CIC-MalMem-2022** (~58.6k amostras, 55 features, tabular puro) |
| Tarefa | **Binária** (benign=0, malware=1) |
| SLM candidatos | Qwen2.5-1.5B (primário), SmolLM2-1.7B (backup) — escolha final no início da Fase 2 |
| Serialização | `key=value` separado por ` ; `, ordem alfabética, 4 casas decimais |
| Quantização | 4-bit via bitsandbytes (re-roda métricas pra comparar impacto) |
| Configs | YAML centralizado (`configs/experiment.yaml`) |

---

## 1) Trilhas do experimento

### Trilha 1 — Baselines tabulares (Fase 1)

| Modelo | Propósito |
|---|---|
| RF | Baseline clássico forte |
| XGBoost | SOTA tabular |
| MLP | Baseline neural |
| MLP→RF | Reprodução do paper base (ASIANCON 2024) |

### Trilha 2 — SLM embeddings + head (Fase 2, método B)

- SLM frozen (sem fine-tune) gera embedding via **mean pooling do last hidden state**
- Head clássico treina sobre o embedding: **XGBoost > RF > LogReg**
- Variante: re-roda com modelo **4-bit** e compara impacto

### Trilha 3 — SLM benign-only (Fase 3, método A, contribuição principal)

- Fine-tune causal LM com **LoRA** apenas em amostras benignas (next-token prediction)
- Score de anomalia = **mean token NLL** por amostra
- Threshold definido por **FPR-alvo** (0.1%, 1%, 5%) em validação benigna
- Protocolo **unseen-family**: leave-one-family-out para simular zero-day

---

## 2) Representação dos dados

### 2.1 Tabular (baselines)

- Features numéricas: normalização **z-score** (fit no train)
- Categóricas: **one-hot encoding**
- Remove colunas que vazam rótulo: `type`, `family` (e qualquer ID/timestamp)
- Remove features constantes e duplicatas

### 2.2 Texto (para SLM)

**Formato exato:**
```
handles_nhandles=21457.0000 ; ldrmodules_not_in_init=0.0000 ; malfind_ninjections=3.0000 ; pslist_nproc=128.0000 ; ...
```

**Regras:**
- Chaves em **ordem alfabética** (reprodutível sem dependência externa)
- Separador: ` ; ` (espaço-ponto-e-vírgula-espaço)
- Numéricos: **4 casas decimais** fixas
- `max_tokens=512` — se serialização exceder, trunca (monitorar com `check_token_lengths`)
- Exclui mesmas colunas do tabular (`type`, `family`, target)

---

## 3) Splits

### 3.1 Split padrão
- **70/15/15** (train/val/test), estratificado por label
- Usado para todos os modelos (comparação justa)

### 3.2 Unseen-family (generalização)
- **Leave-one-family-out**: para cada família de malware no CIC-MalMem-2022, treina sem ela e testa nela
- Test = amostras benignas do test padrão + **todas** as amostras da família held-out
- Usado apenas no método A (benign-only) — é aqui que o paper se diferencia

---

## 4) Arquiteturas

### 4.1 MLP→RF (paper base)

- **MLP**: 3 camadas densas (256→128→64), ReLU, dropout=0.3
- Treina classificação binária (CrossEntropyLoss), Adam, lr=0.001
- **Embedding** = saída da penúltima camada (vetor 64-dim)
- **RF** (500 árvores, balanced) treina sobre o embedding
- Early stopping: patience=5, monitor=val_loss
- Batch: 256, epochs: até 50

### 4.2 SLM embeddings + head (método B)

- **SLM**: Qwen2.5-1.5B ou SmolLM2-1.7B, **frozen** (sem fine-tune)
- `max_seq_len=512`, `batch_size=16`, `dtype=float16`
- **Embedding**: mean pooling do last hidden state (com attention mask)
- **Heads** (treinar os 3, reportar todos):
  - XGBoost (n_estimators=500, max_depth=6, lr=0.1)
  - RF (n_estimators=500)
  - LogReg (C=1.0)
- **Variante 4-bit**: re-carrega modelo com bitsandbytes 4-bit, re-extrai embeddings, re-treina heads

### 4.3 SLM benign-only (método A)

- **Fine-tune**: LoRA (rank=16, alpha=32, dropout=0.05, targets=q_proj+v_proj)
- **Treino**: apenas amostras benignas, causal LM (next-token prediction)
- lr=2e-4, epochs=3, early stopping patience=2
- **Score**: mean token NLL (CrossEntropyLoss sem redução, média com mask de padding)
- **Threshold**: percentil (1-FPR)*100 dos scores de validação benigna
- FPR-alvo: 0.1%, 1%, 5%

---

## 5) Métricas

### Classificação (reportar mean±std sobre 5 seeds)

| Métrica | Tipo |
|---|---|
| **PR-AUC** | Principal |
| ROC-AUC | Secundária |
| F1 macro | Comparação geral |
| F1 por classe | Detalhe |
| Precision / Recall | Detalhe |
| FPR@TPR=95% | Operacional (SOC) |
| Confusion matrix | Visualização |

### Deploy

| Métrica | Onde |
|---|---|
| Latência por amostra (ms) | Todos os modelos |
| Throughput (amostras/s) | Todos os modelos |
| RAM (MB) | Todos os modelos |
| VRAM (MB) | Modelos GPU |

### Benign-only específico

- TPR (recall) para cada FPR-alvo
- PR-AUC usando score contínuo
- Resultados por família (unseen-family)

---

## 6) Plano de execução

### Fase 1 — Baselines (`scripts/run_baselines.py`)
1. Carregar CIC-MalMem-2022, limpar, binarizar
2. Split padrão (5 seeds)
3. Treinar RF, XGBoost, MLP, MLP→RF
4. Métricas + leakage check

### Fase 2 — SLM embeddings (`scripts/run_slm_embed.py`)
5. Serializar dataset em texto
6. Verificar distribuição de tokens (check_token_lengths)
7. Extrair embeddings (frozen SLM)
8. Treinar XGBoost/RF/LogReg sobre embeddings
9. Re-rodar com modelo 4-bit (`--quantize`)

### Fase 3 — Benign-only (`scripts/run_slm_benign.py`)
10. Fine-tune LoRA em textos benignos
11. Scoring (mean token NLL) no test
12. Thresholds por FPR-alvo
13. Protocolo unseen-family (`--unseen-family`)

### Fase 4 — Avaliação (`scripts/run_evaluation.py`)
14. Tabela comparativa mean±std
15. Delta % vs paper base (MLP→RF)
16. Métricas de deploy
17. Tabelas finais para o paper

---

## 7) Estrutura do repositório

```
hugo/
├── configs/
│   └── experiment.yaml          # YAML centralizado (seeds, splits, hiperparâmetros, paths)
├── src/
│   ├── data.py                  # load, clean, split (standard + unseen-family)
│   ├── features.py              # preprocessing tabular + serialização texto
│   ├── baselines.py             # RF, XGBoost, MLP, MLP→RF
│   ├── slm.py                   # embeddings + benign-only scoring + quantização
│   ├── metrics.py               # PR-AUC, F1, FPR@TPR, latência, memória
│   └── utils.py                 # seeds, logging, I/O de configs e resultados
├── scripts/
│   ├── run_baselines.py         # Fase 1
│   ├── run_slm_embed.py         # Fase 2 (aceita --quantize)
│   ├── run_slm_benign.py        # Fase 3 (aceita --unseen-family)
│   └── run_evaluation.py        # Fase 4
├── notebooks/                   # EDA e debug (não é código de produção)
├── outputs/                     # gitignored — modelos, resultados, plots, logs
├── docs/
│   ├── BASELINE.md              # Análise do paper base
│   └── exemplo.md               # Este documento
└── .gitignore
```

---

## 8) Seeds

```
[42, 123, 456, 789, 2024]
```

Todas as operações com aleatoriedade (splits, treino, inicialização) usam a seed corrente. Resultados são agregados como mean±std.

---

## 9) Como rodar

```bash
# Fase 1
python scripts/run_baselines.py --config configs/experiment.yaml

# Fase 2
python scripts/run_slm_embed.py --config configs/experiment.yaml
python scripts/run_slm_embed.py --config configs/experiment.yaml --quantize

# Fase 3
python scripts/run_slm_benign.py --config configs/experiment.yaml
python scripts/run_slm_benign.py --config configs/experiment.yaml --unseen-family

# Fase 4
python scripts/run_evaluation.py --config configs/experiment.yaml
```

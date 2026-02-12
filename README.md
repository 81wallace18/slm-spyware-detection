# Hugo — SLM-first Spyware Detection

Framework de detecção de spyware baseado em Small Language Models (SLMs), com reprodução do baseline ANN→RF (ASIANCON 2024) e dois métodos propostos: embeddings + head clássico e detecção benign-only (zero-day).

Dataset: **CIC-MalMem-2022** | Tarefa: **binária** (benign vs malware)

---

## Estrutura

```
hugo/
├── configs/
│   └── experiment.yaml       # config centralizado (dataset, splits, hiperparâmetros, seeds)
├── src/
│   ├── data.py               # loading, cleaning, splits (standard + unseen-family)
│   ├── features.py           # preprocessing tabular + serialização texto
│   ├── baselines.py          # RF, XGBoost, MLP, MLP→RF
│   ├── slm.py                # SLM embeddings, benign-only scoring, quantização 4-bit
│   ├── metrics.py            # PR-AUC, F1, FPR@TPR, latência, memória
│   └── utils.py              # seeds, logging, I/O
├── scripts/
│   ├── run_baselines.py      # Fase 1
│   ├── run_slm_embed.py      # Fase 2
│   ├── run_slm_benign.py     # Fase 3
│   └── run_evaluation.py     # Fase 4
├── docs/                     # documentação do projeto
├── notebooks/                # EDA e debug
└── outputs/                  # resultados, modelos, logs (gitignored)
```

## Métodos

| Trilha | Descrição | Script |
|---|---|---|
| **Baselines** | RF, XGBoost, MLP, MLP→RF (paper base) | `run_baselines.py` |
| **Método B** | SLM frozen → embedding → XGBoost/RF/LogReg | `run_slm_embed.py` |
| **Método A** | SLM fine-tuned apenas em benigno → anomaly score (zero-day) | `run_slm_benign.py` |

## Requisitos

```
torch
transformers
peft
bitsandbytes
xgboost
scikit-learn
pandas
numpy
psutil
pyyaml
```

## Como rodar

Coloque o CSV do CIC-MalMem-2022 no path definido em `configs/experiment.yaml` e execute as fases em ordem:

```bash
# Fase 1 — Baselines tabulares
python scripts/run_baselines.py

# Fase 2 — SLM embeddings + heads
python scripts/run_slm_embed.py
python scripts/run_slm_embed.py --quantize    # variante 4-bit

# Fase 3 — Benign-only (zero-day)
python scripts/run_slm_benign.py
python scripts/run_slm_benign.py --unseen-family

# Fase 4 — Tabelas comparativas
python scripts/run_evaluation.py
```

Todos os scripts aceitam `--config path/to/config.yaml` (default: `configs/experiment.yaml`).

## Métricas reportadas

- **PR-AUC** (principal), ROC-AUC, F1 macro, F1 por classe
- **FPR@TPR=95%** (operacional para SOC)
- Confusion matrix, precision, recall
- Latência (ms), throughput (amostras/s), RAM/VRAM
- Resultados agregados como **mean±std** sobre 5 seeds

## Reprodutibilidade

- Seeds fixas: `[42, 123, 456, 789, 2024]`
- Splits estratificados determinísticos
- Todos os hiperparâmetros centralizados no YAML
- Curvas de treino (loss/accuracy por epoch) salvas para MLP e MLP→RF

# Como Rodar

Execute os comandos a partir da raiz do projeto:

```bash
cd /home/wallace/Projects/research/slm-spyware-detection-jose
```

## Dependências

```bash
pip install torch transformers peft bitsandbytes accelerate xgboost scikit-learn pandas numpy psutil pyyaml datasets
```

## Execução completa

```bash
python3 dataset.py
python3 scripts/run_baselines.py
python3 scripts/run_slm_embed.py --profile 16gb
python3 scripts/run_slm_benign.py --profile 16gb
python3 scripts/run_evaluation.py
```

## Fases

| Fase | Comando | O que faz |
|---|---|---|
| Dataset | `python3 dataset.py` | Baixa o CIC-MalMem-2022 e salva em `data/CIC-MalMem-2022.csv`. |
| Fase 1 | `python3 scripts/run_baselines.py` | Roda RF, XGBoost, MLP e MLP->RF. |
| Fase 2 | `python3 scripts/run_slm_embed.py --profile 16gb` | Extrai embeddings do SLM em modo 16GB e treina heads clássicos. |
| Fase 3 | `python3 scripts/run_slm_benign.py --profile 16gb` | Fine-tuna o SLM só com benignos e detecta malware por anomalia. |
| Fase 4 | `python3 scripts/run_evaluation.py` | Agrega resultados e gera tabelas comparativas. |

## Unseen-family

Para rodar o protocolo zero-day por família:

```bash
python3 scripts/run_slm_benign.py --profile 16gb --unseen-family
```

## Saídas

Resultados, logs e modelos ficam em:

```text
outputs/
outputs/results/
outputs/logs/
outputs/models/
```

## Observações de VRAM

Use `--profile 16gb` nas fases SLM para reduzir uso de memória. Esse profile usa batch pequeno, quantização 4-bit, NF4 e double quantization.

Se ainda estourar VRAM, reduza `slm.max_seq_len` em `configs/profiles/16gb.yaml` de `512` para `256`.

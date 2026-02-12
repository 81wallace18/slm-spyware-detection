According to a document from **Aug 2024 (ASIANCON 2024)**, o paper base é um híbrido **ANN → Random Forest** para detecção de spyware e reporta principalmente **loss/val_accuracy** como evidência experimental.

A minha recomendação é você construir **um paper novo “SLM-first”** que:

1. **reproduz o baseline ANN→RF** (best-effort) para comparação direta, e
2. **propõe um método centrado em SLM** com protocolos e métricas que hoje são cobrados no estado da arte.

---

# Paper base (para você reproduzir e comparar)

## Metodologia (baseline)

* Pipeline descrito: **ANN** (input/middle/output) gera uma saída e isso “vai para” o **Random Forest**, que toma a decisão final.

## Métricas (como ele reporta)

* Curvas e texto sobre **loss**, **val_accuracy**, **val_loss**; menciona val_accuracy no epoch 10 (80%/87% em trechos diferentes).

## Ponto fraco útil pra você

* Ele próprio diz que accuracy pode não ser ideal em dados desbalanceados e cita precision/recall/F1/AUC, mas **não apresenta isso como resultado principal**.

✅ **O que você faz aqui:** implementa **MLP→RF** (ANN→RF), **RF puro**, **MLP puro** com o *mesmo split* e as *mesmas seeds*.

---

# Nosso paper (recomendação fechada, “SLM-first”)

## Cenário recomendado (dataset)

Use **CIC-MalMem-2022** como cenário principal porque:

* é oficial e bem descrito
* é **50% benign / 50% malicious**
* tem categorias e famílias (inclui Spyware) e dá pra fazer *unseen-family*

(“Spyware families” aparecem explícitas na página e você pode criar um protocolo por família.)

---

## O que vamos fazer de diferente (as contribuições que mais “contam” hoje)

### 1) Núcleo SLM (duas variantes no mesmo paper)

**(A) Benign-only / Zero-day (contribuição principal)**

* Treina o SLM **apenas com benigno**
* Detecta spyware como **anomalia** via score do modelo (ex.: NLL/perplexity/ token-loss)
* Vantagem: narrativa forte para “ameaça nova”, que o paper base só afirma mas não demonstra.

**(B) SLM embeddings + head clássico (contribuição de performance)**

* Serializa cada amostra em texto (“feature=value …”)
* SLM gera **embedding**
* Head leve decide (**XGBoost / RF / Logistic**)
  Isso é um “drop-in” moderno do ANN→RF, ótimo pra bater baseline.

> Isso é alinhado ao estado da arte porque surveys recentes destacam LLMs/SLMs como tendência para melhorar frameworks de detecção e análise, especialmente via representações e interpretabilidade em logs/dados complexos. ([ScienceDirect][1])

---

## Metodologia (como fica seu desenho experimental)

1. **Representações**

* Tabular (features originais) — para baselines clássicos
* Texto (serialização controlada) — para SLM

2. **Baselines obrigatórios (além do paper base)**

* RF puro
* XGBoost/LightGBM (se você quiser um baseline forte e comum)
* MLP puro
* **MLP→RF (paper base)**

3. **Nosso método**

* SLM (embeddings + head)
* SLM (benign-only scoring)

4. **Generalização (o que te coloca “acima” do paper base)**

* **Unseen-family**: treina sem algumas famílias de spyware e testa nessas famílias (zero-day por família).
  O CIC-MalMem-2022 descreve famílias de spyware e isso viabiliza esse protocolo de forma bem defendível. ([unb.ca][2])

> Se você quiser ainda mais “SOTA”, dá pra incluir um baseline Transformer “enxuto” em sequências (quando aplicável), que é justamente uma linha defendida como alternativa a LLMs grandes. ([arXiv][3])

---

## Métricas (pra comparar “X% melhor” com credibilidade)

Use como principais (e reporte média±desvio em 5 seeds):

* **PR-AUC**
* **F1 (macro e por classe)**
* **FPR@TPR=95%** (ou outro TPR fixo)

- matriz de confusão

E inclua **métricas de implantação**:

* latência por amostra, throughput, RAM/VRAM
* versão quantizada (ex.: 4-bit) do SLM (mesmo que seja só no apêndice)

---

## Contribuição (como eu escreveria em 2–3 bullets no abstract)

1. Um framework **SLM-first** para detecção de spyware com **serialização feature→texto** e duas vias: **benign-only** (zero-day) e **embeddings+head** (alta performance).
2. Protocolo de avaliação com **unseen-family** e métricas adequadas a SOC (PR-AUC/FPR), indo além de accuracy/loss.
3. Estudo de **custo de deploy** (quantização/latência), tornando o método aplicável.

---

## Como encaixar no seu hardware (recomendação prática)

* **HPC (muita RAM/CPU):** pré-processamento, geração de splits (incluindo unseen-family), grid-search de RF/XGBoost, rodar 5 seeds.
* **2×24GB VRAM:** fine-tuning leve (LoRA/QLoRA) *se você optar*, e inferência/embeddings do SLM.

---

## Recomendação final (decisão)

Se você quer um paper **bem diferente do base e alinhado ao estado da arte**, faça:

* **Principal:** SLM benign-only (zero-day)
* **Secundário (pra “ganhar em %”):** SLM embeddings + XGBoost/RF
* **Comparação:** reproduz MLP→RF + baselines clássicos, com PR-AUC/FPR/F1 + generalização por família.

Se você disser só **qual tarefa você quer cravar** no CIC-MalMem-2022 (binário *benign vs spyware* OU multiclass), eu já te devolvo a **tabela de experimentos** (baselines × splits × métricas) pronta pra seção “Experimental Setup”.

[1]: https://www.sciencedirect.com/science/article/abs/pii/S0167404824003213?utm_source=chatgpt.com "A survey of large language models for cyber threat detection"
[2]: https://www.unb.ca/cic/datasets/malmem-2022.html "Malware Memory Analysis | Datasets | Canadian Institute for Cybersecurity | UNB"
[3]: https://arxiv.org/html/2408.02313v1?utm_source=chatgpt.com "A Lean Transformer Model for Dynamic Malware Analysis ..."

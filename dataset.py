import pandas as pd
from datasets import load_dataset

# Baixa o dataset do Hugging Face
ds = load_dataset("bvk/CIC-MalMem-2022")

# Converte para DataFrame do Pandas (geralmente está no split 'train')
df = pd.DataFrame(ds['train'])

# Salva no caminho que seu projeto usa
df.to_csv("data/CIC-MalMem-2022.csv", index=False)

print(f"Dataset salvo com sucesso! Total de registros: {len(df)}")
print(f"Colunas disponíveis: {df.columns.tolist()}")
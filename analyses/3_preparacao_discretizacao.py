# Discretização do Addiction_Level
# Autor: Igor Chagas
# Data: 15/10/2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from src.etl import carregar_dataset_bruto
from src.formatting import montar_cabecalho

# Carregando o dataset
df = carregar_dataset_bruto('../data/teen_phone_addiction_dataset.csv')

# Análise
var_analise = 'Addiction_Level'
montar_cabecalho(f"ANÁLISE DA VARIÁVEL {var_analise.upper()}")

# Descritiva
print(f"Mínimo: {df[var_analise].min()}")
print(f"Máximo: {df[var_analise].max()}")
print(f"Média: {df[var_analise].mean():.2f}")
print(f"Mediana: {df[var_analise].median():.2f}")
print(f"Valores únicos: {df[var_analise].nunique()}")

# Distribuição
print("\nDistribuição dos valores:")
print(df[var_analise].describe())

# Criando categorias
montar_cabecalho(f"CRIANDO CATEGORIAS DE {var_analise}")

df['Addiction_Category'] = pd.qcut(
    df[var_analise],
    q=5,
    duplicates='drop'
)

df['Addiction_Category'] = df['Addiction_Category'].cat.codes + 1

# Distribuição por categorias
print("Distribuição por categorias:")
print(df.groupby('Addiction_Category')['Addiction_Level'].describe())

# Visualização
distribuicao = df['Addiction_Category'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')
distribuicao.plot(kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'])
plt.title('Distribuição das categorias de vício', fontsize=16, fontweight='bold')
plt.xlabel('Categoria de vício', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.xticks([0, 1, 2], ['Baixo (1)', 'Moderado (2)', 'Alto (3)'], rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/03_distribuicao_categorias.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*70)

# Salvando
df.to_csv('../data/discretizacao_teen_phone_addiction.csv', index=False)
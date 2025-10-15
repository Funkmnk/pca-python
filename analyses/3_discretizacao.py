# Discretização do Addiction_Level
# Autor: Igor Chagas
# Data: 14/10/2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregando o dataset
df = pd.read_csv('../data/teen_phone_addiction_dataset.csv')
df.columns = df.columns.str.strip()

# Análise
var_analise = 'Addiction_Level'
print("=" * 70)
print(f"ANÁLISE DA VARIÁVEL {var_analise.upper()}")
print("=" * 70)

# Descritiva
print(f"\nMínimo: {df[var_analise].min()}")
print(f"Máximo: {df[var_analise].max()}")
print(f"Média: {df[var_analise].mean():.2f}")
print(f"Mediana: {df[var_analise].median():.2f}")
print(f"Valores únicos: {df[var_analise].nunique()}")

# Distribuição
print("\nDistribuição dos valores:")
print(df[var_analise].describe())

# Criando categorias
print("\n" + "=" * 70)
print(f"CRIANDO CATEGORIAS DE {var_analise}")
print("=" * 70)

df['Addiction_Category'] = pd.qcut(df[var_analise],
                                   q=5,
                                   duplicates='drop'
                                   )

df['Addiction_Category'] = df['Addiction_Category'].cat.codes + 1

# Distribuição por categorias
print("\nDistribuição por categorias:")
print(df.groupby('Addiction_Category')['Addiction_Level'].describe())

df['Addiction_Category'].value_counts().sort_index().plot(kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'])
plt.title('Distribuição das Categorias de Vício')
plt.show()
print("=" * 70)

# Salvando
df.to_csv('../data/discretizacao_teen_phone_addiction.csv')
# Aplicação do LDA - Vício em Celulares
# Autor: Igor Chagas
# Data: 14/10/2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Config visual
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
def montar_cabecalho(texto):
	print("\n" + "=" * 70)
	print(" " * ((70 - len(texto)) // 2) + texto)
	print("=" * 70)

montar_cabecalho("APLICAÇÃO DO LDA")

# Carregando dataset com categorias
df = pd.read_csv('../data/discretizacao_teen_phone_addiction.csv', index_col=0)
df.columns = df.columns.str.strip()
print(f"Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")

# Análise da var alvo
var_alvo = 'Addiction_Category'
montar_cabecalho(f"ANÁLISE DA VARIÁVEL ALVO: {var_alvo}")

print("\nDistribuição das categorias:")
distribuicao = df[var_alvo].value_counts().sort_index()
print(distribuicao)

# Plotando a distribuição
plt.figure(figsize=(10, 6))
distribuicao.plot(kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'])
plt.title('Distribuição das Categorias de Vício', fontsize=16, fontweight='bold')
plt.xlabel('Categoria de Vício', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.xticks([0, 1, 2], ['Baixo (1)', 'Moderado (2)', 'Alto (3)'], rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
# plt.savefig('../plot/lda_01_distribuicao_categorias.png', dpi=300, bbox_inches='tight')
plt.show()

# Selecionando as FEATURES numéricas
montar_cabecalho("SELEÇÃO DE FEATURES PARA O LDA")
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in colunas_numericas if col not in ['ID', 'Addiction_Level', 'Addiction_Category']]

print(f"\nTotal de features selecionadas: {len(features)}")
print("Features:")
for i, feature in enumerate(features, 1):
    print(f"  {i:2d}. {feature}")
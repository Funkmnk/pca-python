# Análise de Correlação - Pearson
# Autor: Igor Chagas
# Data: 14/10/2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from src.etl import carregar_dataset_bruto, obter_features_numericas, exibir_resumo_dataset
from src.formatting import montar_cabecalho

# Config visuais
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Sanitização/limpando espaços
montar_cabecalho("CARREGANDO E SANITIZANDO OS DADOS")
df = carregar_dataset_bruto('../data/teen_phone_addiction_dataset.csv')

# Exibir resumo completo do dataset
exibir_resumo_dataset(df, "INFORMAÇÕES GERAIS DO DATASET")

# Selecionando features numéricas de forma segura
montar_cabecalho("SELEÇÃO DE COLUNAS NUMÉRICAS")
df_numericas = df.select_dtypes(include=[np.number])
print(f"Total de variáveis numéricas: {df_numericas.shape[1]}")
print(f"Variáveis: {list(df_numericas.columns)}\n")

# Matriz de correlação
montar_cabecalho("MATRIZ DE CORRELAÇÃO")
matriz_correlacao = df_numericas.corr(method='pearson')
print(f"Dimensão da matriz: {matriz_correlacao.shape}")

# Visualização - Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(
    matriz_correlacao,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5, 
    cbar_kws={"shrink": 0.8} 
)
plt.title('Matriz de correlação - Dataset vício em celulares', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../plots/01_correlacao_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Clustermap (variáveis agrupadas)
sns.clustermap(
    matriz_correlacao,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    figsize=(14, 10),
    cbar_kws={"shrink": 0.8}
)
plt.suptitle('Clustermap - Agrupamento de variáveis', 
             fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig('../plots/01_correlacao_clustermap.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise de correlação
def encontrar_correlacoes_fortes(matriz_corr, limiar=0.7):
    """
    Identifica correlações fortes na matriz (valores únicos).
    
    Args:
        matriz_corr: Matriz de correlação
        limiar: Valor absoluto mínimo para considerar correlação forte
        
    Returns:
        DataFrame com pares de variáveis e suas correlações
    """
    matriz_upper = matriz_corr.where(
        np.triu(np.ones(matriz_corr.shape), k=1).astype(bool)
    )
    
    correlacoes_fortes = (
        matriz_upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'Variável 1', 
                        'level_1': 'Variável 2', 
                        0: 'Correlação'})
    )
    
    correlacoes_fortes = correlacoes_fortes[
        abs(correlacoes_fortes['Correlação']) >= limiar
    ].sort_values('Correlação', key=abs, ascending=False)
   
    return correlacoes_fortes

correlacoes_fortes = encontrar_correlacoes_fortes(matriz_correlacao, limiar=0.3)

montar_cabecalho("CORRELAÇÕES FORTES (|r| >= 0.3)")
print(correlacoes_fortes)

# Análise de multicolinearidade
correlacoes_muito_fortes = encontrar_correlacoes_fortes(matriz_correlacao, limiar=0.9)

montar_cabecalho("ANÁLISE DE MULTICOLINEARIDADE (|r| >= 0.9)")
if len(correlacoes_muito_fortes) > 0:
    print(correlacoes_muito_fortes)
else:
    print("Nenhuma multicolinearidade severa detectada!")
    
# Salvando as correlações
matriz_correlacao.to_csv('../data/correlacao_matriz_correlacao.csv')
correlacoes_fortes.to_csv('../data/correlacao_correlacoes_fortes.csv', index=False)
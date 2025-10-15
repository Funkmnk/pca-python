import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Config de variáveis visuais
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Carregando o dataset
df = pd.read_csv('../data/teen_phone_addiction_dataset.csv')
df.columns = df.columns.str.strip()

# Visualização inicial
print("=" * 50)
print("INFORMAÇÕES GERAIS DO DATASET")
print("=" * 50)
print(f"\nDimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
print(f"\nPrimeiras linhas:")
print(df.head())
print("-" * 150)
print(f"\nTipos de dados:")
print(df.dtypes)
print("\n DICA: Fique atento a tipos de dados 'object'")
print("-" * 150)

# Análise de consistência
print("\n" + "=" * 50)
print("ANÁLISE DE AUSENTES")
print("=" * 50)
percentual_ausentes = (df.isnull().sum() / len(df)) * 100
print("\nPor coluna (%):")
print(f"{percentual_ausentes.sort_values(ascending=False)}\n")
linhas_ausentes = df[df.isnull().any(axis=1)]
if linhas_ausentes.empty:
    print("Por linha: não há linhas com valores ausentes.")
else:
    pd.set_option('display.width', 200)
    pd.set_option('display.max.columns', None)
    print("Linhas com valores ausentes:")
    print(linhas_ausentes)
    
# Selecionando as colunas numéricas para PCA
print("\n" + "=" * 50)
print("SELEÇÃO DE COLUNAS NUMÉRICAS")
print("=" * 50)
df_numericas = df.select_dtypes(include=[np.number])
print(f"\nTotal de variáveis numéricas: {df_numericas.shape[1]}")
print(f"Variáveis: {list(df_numericas.columns)}\n")
print("-" * 150)

# Matriz de correlação
print("\n" + "=" * 50)
print("MATRIZ DE CORRELAÇÃO")
print("=" * 50)
matriz_correlacao = df_numericas.corr(method='pearson')
print(f"\nDimensão da matriz: {matriz_correlacao.shape}")

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
plt.title('Matriz de Correlação - Dataset Vício em Celulares', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
# plt.savefig('../plot/correlacao_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("-" * 150)

# Clustermap (variáveis agrupadas)
plt.figure(figsize=(14, 10))

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

plt.suptitle('Clustermap - Agrupamento de Variáveis', 
             fontsize=16, y=0.995)
# plt.savefig('../plot/correlacao_clustermap.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise de correlação
def encontrar_corerlacoes_fortes(matriz_corr, limiar=0.7):
    
    matriz_upper = matriz_corr.where(
        np.triu(np.ones(matriz_corr.shape), k=1).astype(bool)
    )
    
    corerlacoes_fortes = (
        matriz_upper.stack().reset_index().rename(columns={'level_0': 'Variável 1', 
                                                            'level_1': 'Variável 2', 
                                                            0: 'Correlação'})
    )
    
    correlacoes_fortes = corerlacoes_fortes[abs(corerlacoes_fortes['Correlação']) >= limiar].sort_values('Correlação', key=abs, ascending=False)
   
    return correlacoes_fortes

correlacoes_fortes = encontrar_corerlacoes_fortes(matriz_correlacao, limiar=0.6) # Definição do limiar de correlação

print("\n" + "=" * 50)
print("CORRELAÇÕES FORTES (|r|)")
print("=" * 50)
print(correlacoes_fortes)
print("-" * 150)

# Análise de correlação (multicolinearidade)
correlacoes_muito_fortes = encontrar_corerlacoes_fortes(matriz_correlacao, limiar=0.9)

print("\n" + "=" * 50)
print("ANÁLISE DE MULTICOLINEARIDADE (|r| >= 0.9)")
print("=" * 50)
if len(correlacoes_muito_fortes) > 0:
    print(correlacoes_muito_fortes)
    print("\nDICA: Considere remover uma das variáveis de cada par antes do PCA")
else:
    print("\nNenhuma multicolinearidade severa detectada!")
    
# Salvando as correlações
matriz_correlacao.to_csv('../data/correlacao_matriz_correlacao.csv')
correlacoes_fortes.to_csv('../data/correlacao_correlacoes_fortes.csv', index=False)
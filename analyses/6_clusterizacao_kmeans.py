# CLUSTERING (K-MEANS) nos componentes LDA
# Autor: Igor Chagas
# Data: 18/10/2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score
from scipy.stats import f_oneway
import sys
sys.path.append('..') # Diretórios
import pickle
from src.etl import carregar_dataset_processado, carregar_dataset_bruto
from src.formatting import montar_cabecalho, montar_divisor
from src.stats_tools import interpretar_silhouette, interpretar_pvalor
from src.visualization import (
    plotar_boxplots_clusters,
    plotar_barras_medias_clusters,
    plotar_heatmap_clusters,
    plotar_radar_chart_clusters,
    visualizar_padronizacao
)

# Config visual
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

montar_cabecalho("CLUSTERING (K-MEANS) NOS COMPONENTES LDA")

# Df de componentes
df_lda = carregar_dataset_processado('../data/lda_componentes.csv')
print(f"Componentes LDA carregados")
print(f"  - Dimensões: {df_lda.shape}")
print(f"  - Colunas: {list(df_lda.columns)}")

# Preparação
montar_cabecalho("PREPARAÇÃO DOS DADOS")

X_clustering_1D = df_lda[['LD1']].values
print(f"Clustering em 1D (LD1):")
print(f"  - Shape: {X_clustering_1D.shape}")
print(f"  - Motivo: LD1 explica 99.7% da separação")

# Padronizando com StandardScaler
scaler = StandardScaler()
X_clustering_1D_scaled = scaler.fit_transform(X_clustering_1D)

print(f"\nEstatísticas da padronização:")
print(f"  ANTES:  μ = {X_clustering_1D.mean():.4f}, σ = {X_clustering_1D.std():.4f}")
print(f"  DEPOIS: μ = {X_clustering_1D_scaled.mean():.4f}, σ = {X_clustering_1D_scaled.std():.4f}")

# Dados para clusterização 
X_clustering = X_clustering_1D_scaled
n_dims = 1
print(f"\nUsando clustering em {n_dims}D")

# Var de comparação
y_true = df_lda['Addiction_Category'].values

# Aplicando Elbow Method
montar_cabecalho("DEFININDO CLUSTERS COM ELBOW METHOD")

k_range = range(2, 11)
inercias = []
silhouette_scores = []

print(f"Testando valores de K de {min(k_range)} a {max(k_range)}...\n")
print("K  |  Inércia  |  Silhouette Score")
print("-" * 40)

for k in k_range:
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_clustering)
    
    inercia = kmeans.inertia_
    silhouette = silhouette_score(X_clustering, clusters)
    
    inercias.append(inercia)
    silhouette_scores.append(silhouette)
    
    print(f"{k}  | {inercia:8.2f} | {silhouette:.4f}")
    
# Plotando o Elbow
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plotando a Inércia
ax1.plot(k_range, inercias, marker='o', linewidth=2, markersize=10, color='steelblue')
ax1.set_xlabel('Número de clusters (k)', fontsize=12)
ax1.set_ylabel('Inércia (Soma das distâncias²)', fontsize=12)
ax1.set_title('Elbow Method - inércia', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.set_xticks(k_range)

# Destacando k=3
k_ideal = 3
idx_ideal = list(k_range).index(k_ideal)
ax1.plot(k_ideal, inercias[idx_ideal], 'ro', markersize=15, label=f'k={k_ideal} (escolhido)')
ax1.legend()

# Plotando o Silhouette Score
ax2.plot(k_range, silhouette_scores, marker='s', linewidth=2, markersize=10, color='darkgreen')
ax2.set_xlabel('Número de clusters (k)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score por k', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.set_xticks(k_range)
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Limiar (0.5)')
ax2.plot(k_ideal, silhouette_scores[idx_ideal], 'ro', markersize=15, label=f'k={k_ideal}')
ax2.legend()

plt.tight_layout()  # BOA PRÁTICA
plt.savefig('../plots/06_kmeans_01_elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

# Explicando escolha
print(f"\nEXPLICANDO K={k_ideal}:")
print(f"  - Inércia em k={k_ideal}: {inercias[idx_ideal]:.2f}")
print(f"  - Silhouette em k={k_ideal}: {silhouette_scores[idx_ideal]:.4f}")
print(f"  - Corresponde às 3 categorias originais de vício")

# Aplicando K-Means
montar_cabecalho(f"APLICANDO K-MEANS COM k={k_ideal}")

kmeans_final = KMeans(n_clusters=k_ideal, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_clustering)

# Clusters no DF
df_lda['Cluster'] = clusters

print("K-Means aplicado com sucesso!")
print("\nDistribuição dos clusters:")
for cluster in sorted(np.unique(clusters)):
    qtd = (clusters == cluster).sum()
    percentual = (qtd / len(clusters)) * 100
    print(f"  - Cluster {cluster}: {qtd:,} indivíduos ({percentual:.1f}%)")

# Centróides
centroides = kmeans_final.cluster_centers_

print(f"\nCentróides dos clusters (LD1 padronizado):")
for i, centro in enumerate(centroides):
    print(f"  - Cluster {i}: LD1 = {centro[0]:.3f}")

# Referência de interpretação
print("\nINTERPRETAÇÃO:")
print("  - Valores negativos: Menor nível de vício")
print("  - Valores próximos de 0: Nível moderado")
print("  - Valores positivos: Maior nível de vício")
    
# Visualizando os clusters
montar_cabecalho("VISUALIZAÇÃO DOS CLUSTERS")

print("Gerando plotagens...")

if n_dims == 1:
    # Histograma + Scatter
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    cores_clusters = {0: '#3498db', 1: '#e67e22', 2: '#e74c3c'}
    labels_clusters = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}
    
    # Histograma
    for cluster in sorted(df_lda['Cluster'].unique()):
        dados_cluster = df_lda[df_lda['Cluster'] == cluster]['LD1']
        ax1.hist(dados_cluster, bins=30, alpha=0.6, 
                label=labels_clusters[cluster], color=cores_clusters[cluster],
                edgecolor='black')
    
    # Centróides
    for i, centro in enumerate(centroides):
        ax1.axvline(x=centro[0], color=cores_clusters[i], 
                   linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Centróide {i}')
    
    ax1.set_xlabel('LD1 (99.7% da separação)', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.set_title('Distribuição dos clusters no eixo LD1', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    # Scatter plot 1D
    np.random.seed(42)
    jitter = np.random.normal(0, 0.1, size=len(df_lda))
    
    for cluster in sorted(df_lda['Cluster'].unique()):
        dados_cluster = df_lda[df_lda['Cluster'] == cluster]
        ax2.scatter(dados_cluster['LD1'], 
                   jitter[df_lda['Cluster'] == cluster],
                   c=cores_clusters[cluster], label=labels_clusters[cluster],
                   s=30, alpha=0.5, edgecolors='black', linewidth=0.3)
    
    # Centróides
    for i, centro in enumerate(centroides):
        ax2.scatter(centro[0], 0, color=cores_clusters[i], 
                   marker='X', s=500, edgecolors='black', linewidth=2,
                   label=f'Centróide {i}', zorder=5)
    
    ax2.set_xlabel('LD1 (99.7% da separação)', fontsize=12)
    ax2.set_ylabel('Jitter (para visualização)', fontsize=12)
    ax2.set_title('Scatter Plot dos clusters', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-1, 1)
    
    plt.tight_layout()  # BOA PRÁTICA
    plt.savefig('../plots/06_kmeans_02_clusters_1d.png', dpi=300, bbox_inches='tight')
    plt.show()

# Silhouette pós cluster
montar_cabecalho("ANÁLISE DE SILHOUETTE POR CLUSTER")

# Score geral
silhouette_avg = silhouette_score(X_clustering, clusters)
print(f"Silhouette Score (geral): {silhouette_avg:.4f}")

# Interpretação
interpretacao_silhouette = interpretar_silhouette(silhouette_avg)
print(f"  -> Qualidade: {interpretacao_silhouette}")

# Silhouette nas amostras
silhouette_vals = silhouette_samples(X_clustering, clusters)

print(f"\nSilhouette Score por cluster:")
for cluster in sorted(np.unique(clusters)):
    cluster_silhouette = silhouette_vals[clusters == cluster]
    print(f"  - Cluster {cluster}: {cluster_silhouette.mean():.4f} " +
          f"(min: {cluster_silhouette.min():.4f}, max: {cluster_silhouette.max():.4f})")
    
# Plotando Silhouette
fig, ax = plt.subplots(figsize=(10, 7))

y_lower = 10
cores_clusters = {0: '#3498db', 1: '#e67e22', 2: '#e74c3c'}

for cluster in sorted(np.unique(clusters)):
    cluster_silhouette = silhouette_vals[clusters == cluster]
    cluster_silhouette.sort()
    
    size_cluster = cluster_silhouette.shape[0]
    y_upper = y_lower + size_cluster
    
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette,
                     facecolor=cores_clusters[cluster],
                     edgecolor=cores_clusters[cluster],
                     alpha=0.7,
                     label=f'Cluster {cluster}')
    
    ax.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster),
            fontsize=12, fontweight='bold')
    
    y_lower = y_upper + 10

# Média geral
ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
          label=f'Média Geral ({silhouette_avg:.3f})')

ax.set_xlabel('Coeficiente Silhouette', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)
ax.set_title('Análise de Silhouette por cluster', fontsize=14, fontweight='bold')
ax.set_yticks([])
ax.set_xlim([-0.2, 1])
ax.legend(loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/06_kmeans_03_silhouette_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Comparando com as categorias originais
montar_cabecalho("COMPARAÇÃO: Clusters vs categorias originais")

# Matriz de confusão
matriz_conf = pd.crosstab(
    df_lda['Addiction_Category'], 
    df_lda['Cluster'], 
    rownames=['Addiction_Category'], 
    colnames=['Cluster']
)

print("Matriz de confusão:")
print(matriz_conf)

# Plotando a matriz
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues', 
           cbar_kws={'label': 'Frequência'})
plt.title('Matriz de confusão: categorias vs clusters', fontsize=14, fontweight='bold')
plt.ylabel('Addiction_Category (original)', fontsize=12)
plt.xlabel('Cluster (K-means)', fontsize=12)
plt.tight_layout()
plt.savefig('../plots/06_kmeans_04_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Métricas
ari = adjusted_rand_score(y_true, clusters)
nmi = normalized_mutual_info_score(y_true, clusters)
homogeniedade = homogeneity_score(y_true, clusters)
integralidade = completeness_score(y_true, clusters)

montar_divisor("Métricas de Concordância", 70)
print(f"  - Adjusted Rand Index (ARI):  {ari:.4f}")
print(f"  - Normalized Mutual Info (NMI): {nmi:.4f}")
print(f"  - Homogeneidade: {homogeniedade:.4f}")
print(f"  - Integralidade: {integralidade:.4f}")
print("\n  (Escala: 0.0 = sem concordância, 1.0 = concordância perfeita)")

# Interpretação
print("\nINTERPRETAÇÃO:")
if ari > 0.8:
    print("  EXCELENTE concordância! Clusters ≈ Categorias")
elif ari > 0.6:
    print("  BOA concordância! Clusters similares às categorias")
elif ari > 0.4:
    print("  Concordância MODERADA. Clusters apontam novos padrões")
else:
    print("  BAIXA concordância. Clusters muito diferentes das categorias")


#                         CARACTERIZAÇÃO DOS CLUSTERS
montar_cabecalho("CARACTERIZAÇÃO DOS CLUSTERS")

# Carregando df original (descritiva)
df_original = carregar_dataset_bruto('../data/teen_phone_addiction_dataset.csv')

print("Dataset original carregado para caracterização")
print(f"  - Shape: {df_original.shape}")

# Juntando os clusters ao dataset
df_original = df_original.reset_index(drop=True)
df_lda = df_lda.reset_index(drop=True)
df_original['Cluster'] = df_lda['Cluster'].values

print("\nDistribuição final dos clusters:")
for cluster in sorted(df_original['Cluster'].unique()):
    qtd = (df_original['Cluster'] == cluster).sum()
    percentual = (qtd / len(df_original)) * 100
    print(f"  - Cluster {cluster}: {qtd:,} ({percentual:.1f}%)")

# Escolhendo as variáveis de descrição
montar_cabecalho("SELEÇÃO DAS VARIÁVEIS IMPORTANTES")

variaveis_numericas = [
    'Age',
    'Daily_Usage_Hours',
    'Sleep_Hours',
    'Academic_Performance',
    'Social_Interactions',
    'Anxiety_Level',
    'Depression_Level',
    'Self_Esteem',
    'Parental_Control',
    'Phone_Checks_Per_Day',
    'Time_on_Social_Media'
]

print(f"Selecionadas {len(variaveis_numericas)} variáveis numéricas:")
for var in variaveis_numericas:
    print(f"  - {var}")

# Estatísticas descritivas por cluster
montar_cabecalho("ESTATÍSTICAS DESCRITIVAS POR CLUSTER")

# Média dos clusters
medias_por_cluster = df_original.groupby('Cluster')[variaveis_numericas].mean().round(2)

montar_divisor("MÉDIAS POR CLUSTER", 70)
print(medias_por_cluster.T.to_string())

# Visualização
montar_cabecalho("VISUALIZAÇÃO GRÁFICA")

print("Boxplots...")
plotar_boxplots_clusters(df_original, variaveis_numericas)

print("Gráfico de médias...")
plotar_barras_medias_clusters(medias_por_cluster, variaveis_numericas)

print("Heatmap...")
plotar_heatmap_clusters(medias_por_cluster)

print("Radar chart...")
plotar_radar_chart_clusters(medias_por_cluster, variaveis_numericas)

# Testes estatísticos - ANOVA
montar_cabecalho("TESTE ANOVA - DIFERENÇAS ENTRE CLUSTERS")

print("H₀: As médias dos clusters são iguais")
print("H₁: Pelo menos uma média é diferente")
print("Significância: α = 0.05\n")

resultados_anova = []

for var in variaveis_numericas:
    # Grupos
    grupos = [df_original[df_original['Cluster'] == c][var].values 
              for c in sorted(df_original['Cluster'].unique())]
    
    f_stat, p_value = f_oneway(*grupos)
    
    # Interpretação
    interpretacao = interpretar_pvalor(p_value, alpha=0.05)
    
    print(f"{var}:")
    print(f"  - F-statistic: {f_stat:.4f}")
    print(f"  - P-valor: {p_value:.6f}")
    print(f"  - Resultado: {interpretacao['nivel']}\n")
    
    resultados_anova.append({
        'Variavel': var,
        'F_statistic': f_stat,
        'P_valor': p_value,
        'Nivel': interpretacao['nivel']
    })

# DF com resultados
df_anova = pd.DataFrame(resultados_anova)
df_anova = df_anova.sort_values('P_valor')

montar_divisor("Variáveis Organizadas por Significância", 70)
print(df_anova.to_string(index=False))

# Salvando relatórios
df_anova.to_csv('../data/clusterizacao_anova.csv', index=False)
medias_por_cluster.to_csv('../data/clusterizacao_caracterizacao_medias.csv')
df_original.to_csv('../data/clusterizacao_dataset_com_clusters.csv', index=False)

# Salvando padronizador
with open('../models/scaler_kmeans.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Salvar modelo (K-Means)
with open('../models/kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Salvando perfis dos clusters
perfis_clusters = {
    0: "Uso Controlado - Baixo risco de vício",
    1: "Uso Intenso - Alto risco de vício",
    2: "Uso Moderado - Risco intermediário"
}
with open('../models/perfis_clusters.pkl', 'wb') as f:
    pickle.dump(perfis_clusters, f)

# Interpretação final
montar_cabecalho("CARACTERIZAÇÃO FINAL DOS CLUSTERS")

# Ordenando vars por significativas
vars_sig = df_anova[df_anova['P_valor'] < 0.05]['Variavel'].tolist()

# Cluster 0
montar_divisor("Cluster 0 - Uso Controlado", 70)
qtd_c0 = (df_original['Cluster'] == 0).sum()
pct_c0 = (qtd_c0 / len(df_original)) * 100
print(f"Tamanho: {qtd_c0:,} adolescentes ({pct_c0:.1f}%)\n")
print("Características principais (variáveis mais significativas):")
for var in vars_sig[:5]:  # Top 5
    valor = medias_por_cluster.loc[0, var]
    print(f"  - {var}: {valor:.2f}")
print("\nPERFIL: Uso saudável, com baixos níveis de uso de smartphone e melhor qualidade de sono em relação aos outros grupos.")

# Cluster 1
montar_divisor("Cluster 1 - Uso Intenso", 70)
qtd_c1 = (df_original['Cluster'] == 1).sum()
pct_c1 = (qtd_c1 / len(df_original)) * 100
print(f"Tamanho: {qtd_c1:,} adolescentes ({pct_c1:.1f}%)\n")
print("Características principais (variáveis mais significativas):")
for var in vars_sig[:5]:
    valor = medias_por_cluster.loc[1, var]
    print(f"  - {var}: {valor:.2f}")
print("\nPERFIL: Contraponto direto ao cluster 0, com o maior nível de uso, tanto em horas quanto em checagens, e em indicadores chave (ANOVA). Sono prejudicado, tendo o menor valor dos grupos.")

# Cluster 2
montar_divisor("Cluster 2 - Uso Moderado", 70)
qtd_c2 = (df_original['Cluster'] == 2).sum()
pct_c2 = (qtd_c2 / len(df_original)) * 100
print(f"Tamanho: {qtd_c2:,} adolescentes ({pct_c2:.1f}%)\n")
print("Características principais (variáveis mais significativas):")
for var in vars_sig[:5]:
    valor = medias_por_cluster.loc[2, var]
    print(f"  - {var}: {valor:.2f}")
print("\nPERFIL: Maior grupo dos 3, com indicadores de uso controlados, se posicionando entre o Cluster 0 e Cluster 1.")

print("="*70)
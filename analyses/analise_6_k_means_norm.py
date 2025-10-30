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
from utils import montar_cabecalho

# Config visual
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

montar_cabecalho("CLUSTERING (K-MEANS) NOS COMPONENTES LDA")

# Df de componentes
df_lda = pd.read_csv('../data/lda_componentes.csv')
print("Componentes LDA:")
print(f"  Dimensões: {df_lda.shape}")
print(f"  Colunas: {list(df_lda.columns)}")

# Preparação
montar_cabecalho("PREPARAÇÃO DOS DADOS")

X_clustering_1D = df_lda[['LD1']].values
print(f"Clustering em 1D (LD1):")
print(f"  Shape: {X_clustering_1D.shape}")
print(f"  Motivo: LD1 explica 99.7% da separação")

X_clustering = X_clustering_1D
n_dims = 1
print(f"Usando clustering em {n_dims}D")

# Normalização
montar_cabecalho("NORMALIZAÇÃO DOS COMPONENTES LDA")

print("Estatísticas ANTES da normalização:")
print(f"  LD1 - Média: {X_clustering[:, 0].mean():.4f}")
print(f"  LD1 - Desvio Padrão: {X_clustering[:, 0].std():.4f}")
print(f"  LD1 - Min: {X_clustering[:, 0].min():.4f}")
print(f"  LD1 - Max: {X_clustering[:, 0].max():.4f}")

# Normalizando
scaler = StandardScaler()
X_clustering_scaled = scaler.fit_transform(X_clustering)

if X_clustering_scaled.ndim == 1:
    X_clustering_scaled = X_clustering_scaled.reshape(-1, 1)

print("\nEstatísticas DEPOIS da normalização:")
print(f"  LD1 - Média: {X_clustering_scaled[:, 0].mean():.4f}")
print(f"  LD1 - Desvio Padrão: {X_clustering_scaled[:, 0].std():.4f}")
print(f"  LD1 - Min: {X_clustering_scaled[:, 0].min():.4f}")
print(f"  LD1 - Max: {X_clustering_scaled[:, 0].max():.4f}")

print("\nDados normalizados: Média = 0, Desvio = 1")

# plotando a visualização
print("\nPlotando a normalização...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

dados_antes = X_clustering.flatten()
dados_depois = X_clustering_scaled.flatten()

# Histogramas
# Antes da normalização
axes[0, 0].hist(dados_antes, bins=40, color='steelblue', 
                alpha=0.7, edgecolor='black')
axes[0, 0].axvline(dados_antes.mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Média = {dados_antes.mean():.2f}')
axes[0, 0].axvline(np.median(dados_antes), color='orange', 
                   linestyle='--', linewidth=2, label=f'Mediana = {np.median(dados_antes):.2f}')
axes[0, 0].set_xlabel('LD1 (escala original)', fontsize=11)
axes[0, 0].set_ylabel('Frequência', fontsize=11)
axes[0, 0].set_title('Antes da normalização - Histograma', fontsize=12, fontweight='bold')
axes[0, 0].legend(loc='best')
axes[0, 0].grid(alpha=0.3)

# Depois da normalização
axes[0, 1].hist(dados_depois, bins=40, color='darkgreen', 
                alpha=0.7, edgecolor='black')
axes[0, 1].axvline(dados_depois.mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Média = {dados_depois.mean():.2f}')
axes[0, 1].axvline(np.median(dados_depois), color='orange', 
                   linestyle='--', linewidth=2, label=f'Mediana = {np.median(dados_depois):.2f}')
axes[0, 1].set_xlabel('LD1 (Normalizado)', fontsize=11)
axes[0, 1].set_ylabel('Frequência', fontsize=11)
axes[0, 1].set_title('Depois da normalização - Histograma', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc='best')
axes[0, 1].grid(alpha=0.3)

# Boxplots
# Antes da normalização
bp1 = axes[1, 0].boxplot(dados_antes, vert=True, patch_artist=True,
                          widths=0.5, showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                          medianprops=dict(color='orange', linewidth=2),
                          boxprops=dict(facecolor='steelblue', alpha=0.7),
                          whiskerprops=dict(color='black', linewidth=1.5),
                          capprops=dict(color='black', linewidth=1.5))
axes[1, 0].set_ylabel('LD1 (Escala Original)', fontsize=11)
axes[1, 0].set_title('Aantes da Normalização - Boxplot', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks([1])
axes[1, 0].set_xticklabels(['LD1'])
axes[1, 0].grid(axis='y', alpha=0.3)

# Estatísticas
q1, median, q3 = np.percentile(dados_antes, [25, 50, 75])
axes[1, 0].text(1.3, q1, f'Q1 = {q1:.2f}', fontsize=9, va='center')
axes[1, 0].text(1.3, median, f'Mediana = {median:.2f}', fontsize=9, va='center')
axes[1, 0].text(1.3, q3, f'Q3 = {q3:.2f}', fontsize=9, va='center')

# Depois da normalização
bp2 = axes[1, 1].boxplot(dados_depois, vert=True, patch_artist=True,
                          widths=0.5, showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                          medianprops=dict(color='orange', linewidth=2),
                          boxprops=dict(facecolor='darkgreen', alpha=0.7),
                          whiskerprops=dict(color='black', linewidth=1.5),
                          capprops=dict(color='black', linewidth=1.5))
axes[1, 1].set_ylabel('LD1 (Normalizado)', fontsize=11)
axes[1, 1].set_title('Depois da normalização - Boxplot', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks([1])
axes[1, 1].set_xticklabels(['LD1'])
axes[1, 1].grid(axis='y', alpha=0.3)

# Estatísticas
q1_norm, median_norm, q3_norm = np.percentile(dados_depois, [25, 50, 75])
axes[1, 1].text(1.3, q1_norm, f'Q1 = {q1_norm:.2f}', fontsize=9, va='center')
axes[1, 1].text(1.3, median_norm, f'Mediana = {median_norm:.2f}', fontsize=9, va='center')
axes[1, 1].text(1.3, q3_norm, f'Q3 = {q3_norm:.2f}', fontsize=9, va='center')

# Linha de referência
axes[1, 1].axhline(y=0, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Zero')
axes[1, 1].legend(loc='best')

plt.suptitle('Impacto da normalização (StandardScaler) nos componentes LDA', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../plot/comparacao/normalizacao_distribuicao.png', dpi=300, bbox_inches='tight')
plt.show()

# Var de comparação
y_true = df_lda['Addiction_Category'].values

# Aplicando Elbow Method
montar_cabecalho("DIFINIDO CLUSTERS COM ELBOW METHOD")

k_range = range(2, 11)
inercias = []
silhouette_scores = []

print(f"Testando valores de K...")
print("\nk  |  Inércia  |  Silhouette Score")
print("-" * 40)

for k in k_range:
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_clustering_scaled)
    
    inercia = kmeans.inertia_
    silhouette = silhouette_score(X_clustering_scaled, clusters)
    
    inercias.append(inercia)
    silhouette_scores.append(silhouette)
    
    print(f"{k}  | {inercia:8.2f} | {silhouette:.4f}")
    
# Plotando o Elbow
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plotando a Inércia
ax1.plot(k_range, inercias, marker='o', linewidth=2, markersize=10, color='steelblue')
ax1.set_xlabel('Número de clusters (k)', fontsize=12)
ax1.set_ylabel('Inércia (Soma das Distâncias²)', fontsize=12)
ax1.set_title('Elbow Method - Inércia', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.set_xticks(k_range)

# Destacando k=3 (= categorias depois da discretização)
k_ideal = 3
idx_ideal = list(k_range).index(k_ideal)
ax1.plot(k_ideal, inercias[idx_ideal], 'ro', markersize=15, label=f'k={k_ideal}')
ax1.legend()

# Plotando o Silhouette Socre
ax2.plot(k_range, silhouette_scores, marker='s', linewidth=2, markersize=10, color='darkgreen')
ax2.set_xlabel('Número de clusters (k)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score por k', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.set_xticks(k_range)
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Limiar (0.5)')

# Checar K=3
ax2.plot(k_ideal, silhouette_scores[idx_ideal], 'ro', markersize=15, label=f'k={k_ideal}')
ax2.legend()

plt.tight_layout()
plt.savefig('../plot/comparacao/kmeans_01_elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

# Aplicando K-Means
montar_cabecalho(f"APLICANDO K-MEANS COM k={k_ideal}")

kmeans_final = KMeans(n_clusters=k_ideal, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_clustering_scaled)

# Clusters no DF
df_lda['Cluster'] = clusters

print("K-Means aplicado!")
print("\nDistribuição dos clusters:")
print(df_lda['Cluster'].value_counts().sort_index())

# Centróides
centroides_scaled = kmeans_final.cluster_centers_
centroides = scaler.inverse_transform(centroides_scaled)

print(f"\nCentróides dos clusters (LD1 - escala original):")
for i, centro in enumerate(centroides):
    print(f"  Cluster {i}: LD1 = {centro[0]:.3f}")
    
# Visualizando os clusters
montar_cabecalho("VISUALIZAÇÃO DOS CLUSTERS")

if n_dims == 1:
    # Histograma
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    cores_clusters = {0: '#3498db', 1: '#e67e22', 2: '#e74c3c'}
    labels_clusters = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}
    
    for cluster in sorted(df_lda['Cluster'].unique()):
        dados_cluster = df_lda[df_lda['Cluster'] == cluster]['LD1']
        ax1.hist(dados_cluster, bins=30, alpha=0.6, 
                label=labels_clusters[cluster], color=cores_clusters[cluster],
                edgecolor='black')
    
    # Centróides
    for i, centro in enumerate(centroides):
        ax1.axvline(x=centro[0], color=cores_clusters[i], 
                   linestyle='--', linewidth=2, alpha=0.8)
    
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
    
    plt.tight_layout()
    plt.savefig('../plot/comparacao/kmeans_02_clusters_1d.png', dpi=300, bbox_inches='tight')
    plt.show()

else:
    # Scatter plot
    plt.figure(figsize=(12, 8))
    
    cores_clusters = {0: '#3498db', 1: '#e67e22', 2: '#e74c3c'}
    labels_clusters = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}
    
    for cluster in sorted(df_lda['Cluster'].unique()):
        dados_cluster = df_lda[df_lda['Cluster'] == cluster]
        plt.scatter(dados_cluster['LD1'], dados_cluster['LD2'],
                   c=cores_clusters[cluster], label=labels_clusters[cluster],
                   s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    plt.scatter(centroides[:, 0], centroides[:, 1],
               marker='X', s=500, c='black', edgecolors='white', linewidth=2,
               label='Centróides', zorder=5)
    
    plt.xlabel('LD1 (99.7% da separação)', fontsize=12)
    plt.ylabel('LD2 (0.3% da separação)', fontsize=12)
    plt.title('Clusters no Espaço LDA (2D)', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig('../plot/comparacao/kmeans_02_clusters_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
# Silhouette pós clusterização
montar_cabecalho("ANÁLISE DE SILHOUETTE POR CLUSTER")

# Score geral
silhouette_avg = silhouette_score(X_clustering_scaled, clusters)
print(f"Silhouette Score (geral): {silhouette_avg:.4f}")

if silhouette_avg > 0.7:
    qualidade = 'EXCELENTE'
elif silhouette_avg > 0.5:
    qualidade = "BOM"
elif silhouette_avg > 0.3:
    qualidade = "RAZOÁVEL"
else:
    qualidade = "RUIM"
    
print(f"  Qualidade dos clusters: {qualidade}")

# Silhouette nas amostras
silhouette_vals = silhouette_samples(X_clustering_scaled, clusters)

print(f"\nSilhouette Score por cluster:")
for cluster in sorted(np.unique(clusters)):
    cluster_silhouette = silhouette_vals[clusters == cluster]
    print(f"   Cluster {cluster}: {cluster_silhouette.mean():.4f} " +
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
plt.savefig('../plot/comparacao/kmeans_03_silhouette_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Comparando com as categorias originais
montar_cabecalho("COMPARAÇÃO: Clusters vs Categorias Originais")

# Matriz de confusão
matriz_conf = pd.crosstab(df_lda['Addiction_Category'], df_lda['Cluster'], rownames=['Addiction_Category'], colnames=['Cluster'])

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
plt.savefig('../plot/comparacao/kmeans_04_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Métricas
ari = adjusted_rand_score(y_true, clusters)
nmi = normalized_mutual_info_score(y_true, clusters)
homogeniedade = homogeneity_score(y_true, clusters)
integralidade = completeness_score(y_true, clusters)

print("\nMétricas de concordância:")
print(f"  Adjusted Rand Index (ARI):  {ari:.4f} (1.0 = perfeito)")
print(f"  Normalized Mutual Info (NMI): {nmi:.4f} (1.0 = perfeito)")
print(f"  Homogeniedade: {homogeniedade:.4f} (1.0 = perfeito)")
print(f"  Integralidade: {integralidade:.4f} (1.0 = perfeito)")

# Interpretação
print(f"\nInterpretação:")
if ari > 0.8:
    print("  EXECELENTE concordância! Clusters ≈ Categorias")
elif ari > 0.6:
    print("  BOA concordância! Clusters similares às categorias")
elif ari > 0.4:
    print("  Concordância MODERADA. Clusters mostram novos padrões")
else:
    print("  BAIXA concordância. Clusters muito diferentes das categorias")
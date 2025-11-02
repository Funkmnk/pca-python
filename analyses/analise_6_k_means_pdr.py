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
from utils import montar_cabecalho, visualizar_padronizacao, montar_divisor
from utils import (plotar_boxplots_clusters, plotar_barras_medias_clusters, plotar_heatmap_clusters, plotar_radar_chart_clusters)
from scipy.stats import f_oneway

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

# Padronizando com StandarScaller
scaler = StandardScaler()
X_clustering_1D_scaled = scaler.fit_transform(X_clustering_1D)

print(f"\nEstatísticas antes da padronização:")
print(f"  Média de LD1: {X_clustering_1D.mean():.4f}")
print(f"  Desvio padrão de LD1: {X_clustering_1D.std():.4f}")

print(f"\nEstatísticas depois da padronização:")
print(f"  Média de LD1_scaled: {X_clustering_1D_scaled.mean():.4f}")
print(f"  Desvio padrão de LD1_scaled: {X_clustering_1D_scaled.std():.4f}")

# visualizar_padronizacao(X_clustering_1D, X_clustering_1D_scaled, 'LD1', '../plot/comparacao/standardscaler_comparacao.png')

# Dados para clusterização 
X_clustering = X_clustering_1D_scaled
n_dims = 1
print(f"\nUsando clustering em {n_dims}D")

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
plt.savefig('../plot/kmeans_01_elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

# Aplicando K-Means
montar_cabecalho(f"APLICANDO K-MEANS COM k={k_ideal}")

kmeans_final = KMeans(n_clusters=k_ideal, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_clustering)

# Clusters no DF
df_lda['Cluster'] = clusters

print("K-Means aplicado!")
print("\nDistribuição dos clusters:")
print(df_lda['Cluster'].value_counts().sort_index())

# Centróides
centroides = kmeans_final.cluster_centers_

print(f"\nCentróides dos clusters (LD1):")
for i, centro in enumerate(centroides):
    print(f"  Cluster {i+1}: LD1 = {centro[0]:.3f}")
    
# Visualizando os clusters
montar_cabecalho("VISUALIZAÇÃO DOS CLUSTERS")

print("Exibindo gráficos...")

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
    plt.savefig('../plot/kmeans_02_clusters_1d.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('../plot/kmeans_02_clusters_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
# Silhouette pós clusterização
montar_cabecalho("ANÁLISE DE SILHOUETTE POR CLUSTER")

# Score geral
silhouette_avg = silhouette_score(X_clustering, clusters)
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
silhouette_vals = silhouette_samples(X_clustering, clusters)

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
plt.savefig('../plot/kmeans_03_silhouette_analysis.png', dpi=300, bbox_inches='tight')
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
plt.savefig('../plot/kmeans_04_confusion_matrix.png', dpi=300, bbox_inches='tight')
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
    
#==========================================================================================
#                             CARACTERIZAÇÃO DOS CLUSTERS
#==========================================================================================
montar_cabecalho("CARACTERIZAÇÃO DOS CLUSTERS")

# Carregando df original (descritiva)
df_original = pd.read_csv('../data/teen_phone_addiction_dataset.csv')
df_original.columns = df_original.columns.str.strip()

print("Carregando dataset original...")
print(f"  Shape do dataset (original): {df_original.shape}")
print(f"  Colunas do dataset ({df_original.shape[1]}):")
for col in df_original.columns:
    print(f"  - {col}")

# Juntando os clusters ao dataset
df_original = df_original.reset_index(drop=True)
df_lda = df_lda.reset_index(drop=True)

df_original['Cluster'] = df_lda['Cluster'].values

print("\nDistribuição dos clusters:")
print(df_original['Cluster'].value_counts().sort_index())

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

variaveis_categoricas = [
    'Gender',
    'Phone_Usage_Purpose',
    'Family_Communication'
]

print(f"\nNuméricas ({len(variaveis_numericas)}):")
for var in variaveis_numericas:
    print(f"  - {var}")

print(f"\nCategóricas ({len(variaveis_categoricas)}):")
for var in variaveis_categoricas:
    print(f"  - {var}")

# Descritia por cluster
montar_cabecalho("ESTATÍSTICAS DESCRITIVAS POR CLUSTER")

estatisticas_numericas = df_original.groupby('Cluster')[variaveis_numericas].agg([
    'count',
    'mean',
    'std',
    'min',
    'median',
    'max'
]).round(2)

print("Estatísticas numércias por cluster:")
for cluster in sorted(df_original['Cluster'].unique()):
    print(f"{montar_divisor(f"Cluster {cluster}", 70)}")
    print(estatisticas_numericas.loc[cluster].to_string())
    
# Média dos clusters
medias_por_cluster = df_original.groupby('Cluster')[variaveis_numericas].mean().round(2)

montar_divisor("MÉDIAS POR CLUSTER", 70)
print(medias_por_cluster.T.to_string())

# Distribuição das variáveis categóricas
montar_divisor("DISTRIBUIÇÃO DE VARIÁVEIS CATEGÓRICAS", 70)

print("-"*70)  
for var in variaveis_categoricas:
    crosstab = pd.crosstab(
        df_original['Cluster'], 
        df_original[var], 
        normalize='index'
    ) * 100
    print(crosstab.round(1).to_string())
    print("-"*70)
    
# Visualização (assim vai)
montar_cabecalho("Visualização gráfica")

# Box plots
print("Plotando boxplots...")
plotar_boxplots_clusters(df_original, variaveis_numericas)

# Medias
print("\nPlotando médias...")
plotar_barras_medias_clusters(medias_por_cluster, variaveis_numericas)

# Heatmap
print("\nPlotando heatmap...")
plotar_heatmap_clusters(medias_por_cluster)

# Spider plot
print("\nPlotando spider plot...")
plotar_radar_chart_clusters(medias_por_cluster, variaveis_numericas)

"""
Cluster 0 tem um perfil de uso menor, com menor tempo de uso, menor tempo em redes sociais e checagem de celular
mais baixa, mas o desempenho acadêmico e idade é menor em relação aos outros grupos. Em contraponto, eles tem maior
nível de interações sociais, níveis de ansiedade e depressão mais baixos que a média, maior auto estima, maior 
presença de controle parental e dorme melhor.

Cluster 1 tem um perfil de uso de celular muito maior (uso diário, tempo em rede social e checagem de celular por dia), 
tendo um perfil menos saudável em relação a sono, níveis de auto estima, depressão, ansiedade e interações sociais, 
a presença parental também é menor, isso pode indicar um lar menos saudável, e que não necessariamente se reflete no 
desempenho acadêmico, sendo ele moderado.

Desempenho acadêmico e saúde mental
>
>Cluster 2 tem um perfil de grande risco, mesmo tendo um uso de celulares próximo do normal, os níveis de ansiedade e 
depressão são os maiores até agora, com níveis de interação social, desempenho acadêmico bem acima da média, e controle
parental baixo, sua saúde mental pode se camuflar. A aparente causa demostra ser o desempenho acadêmico elevado, a pressão 
por ele pode deprimir os adolescentes, que sem a atenção adequada dos pais, apresentam um perfil de risco que necessita de 
atenção imediata.
"""

# Testes estatísticos
# Anova
montar_cabecalho("TESTE ANOVA")
montar_divisor("Resultados:", 40)
print("H0 (Hipótese Nula): As médias dos clusters são iguais")
print("H1 (Hipótese Alternativa): Pelo menos uma média é diferente")
print("Significância: α = 0.05 (p-value < 0.05 → rejeitar H0)\n")

resultados_anova = []

for var in variaveis_numericas:
    # Grupos
    grupos = [df_original[df_original['Cluster'] == c][var].values 
              for c in sorted(df_original['Cluster'].unique())]
    
    f_stat, p_value = f_oneway(*grupos)
    
    # Interpretação
    if p_value < 0.001:
        significancia = "ALTAMENTE SIGNIFICATIVO"
    elif p_value < 0.01:
        significancia = "MUITO SIGNIFICATIVO"
    elif p_value < 0.05:
        significancia = "SIGNIFICATIVO"
    else:
        significancia = "NÃO SIGNIFICATIVO"
        
    resultados_anova.append({
        'Variável': var,
        'F-statistic': f_stat,
        'p-value': p_value,
        'Significância': significancia
    })
    
    print(f"Variável: {var}")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   p-value: {p_value:.6f}")
    print(f"   Resultado: {significancia}")
    print()
    
# DF com resultados
df_anova = pd.DataFrame(resultados_anova)
df_anova = df_anova.sort_values('p-value')

# Apresentando
montar_divisor("Variáveis organizadas por significancia", 40)
print(df_anova.to_string(index=False))

# Interpretação final (desconsiderando o heatmap padronizado, considerando apenas ANOVA)
montar_cabecalho("CARACTERIZANDO OS CLUSTERS")

# Organizando as features
vars_sig = df_anova[df_anova['p-value'] < 0.05]['Variável'].tolist()

# Cluster 0
montar_divisor("Cluster 0 - Uso controlado", 70)
print(f"Tamanho: {(df_original['Cluster'] == 0).sum()} adolescentes " +
      f"({(df_original['Cluster'] == 0).sum() / len(df_original) * 100:.1f}%)")

print("\nCaracterísticas principais (por significância):")
for var in vars_sig:
    valor = medias_por_cluster.loc[0, var]
    print(f"  • {var}: {valor:.2f}")

print("\nInterpretação:")
print("  Uso saudável, com baixos níveis de uso de smartphone e melhor qualidade de sono em relação aos outros grupos.")

# Cluster 1
montar_divisor("Cluster 1 - Uso intenso", 70)
print(f"Tamanho: {(df_original['Cluster'] == 1).sum()} adolescentes " +
      f"({(df_original['Cluster'] == 1).sum() / len(df_original) * 100:.1f}%)")

print("\nCaracterísticas principais (por significância):")
for var in vars_sig:
    valor = medias_por_cluster.loc[1, var]
    print(f"  • {var}: {valor:.2f}")

print("\nInterpretação:")
print("  Contraponto direto ao cluster 0, com o maior nível de uso, tanto em horas quanto em checagens, e em indicadores chave (ANOVA). Sono prejudicado, tendo o menor valor dos grupos")

# Cluster 2
montar_divisor("Cluster 2 - Uso moderado", 70)
print(f"Tamanho: {(df_original['Cluster'] == 2).sum()} adolescentes " +
      f"({(df_original['Cluster'] == 2).sum() / len(df_original) * 100:.1f}%)")

print("\nCaracterísticas principais (por significância):")
for var in vars_sig:
    valor = medias_por_cluster.loc[2, var]
    print(f"  • {var}: {valor:.2f}")

print("\nInterpretação:")
print("  Maior grupo dos 3, com indicadores de uso controlados, se posicionando entre o Cluster 0 e Cluster 1.")

# Clusters
df_original.to_csv('../data/clusterizacao/dataset_com_clusters.csv', index=False)

# Descritiva
medias_por_cluster.to_csv('../data/clusterizacao/caracterizacao_medias.csv')

# ANOVA
df_anova.to_csv('../data/clusterizacao/caracterizacao_anova.csv', index=False)
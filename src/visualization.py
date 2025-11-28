"""
Módulo de Visualização - Funções para geração de gráficos
Autor: Igor Chagas
Data: 26/11/2024

Responsabilidades:
- Criação de gráficos padronizados
- Plots de análise exploratória
- Visualizações de clustering
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def visualizar_padronizacao(X_original, X_padronizado, nome_variavel='Variável', 
                            salvar_em='../plots/standardscaler_comparacao.png'):
    """
    Cria visualização comparativa antes/depois da padronização.
    
    Args:
        X_original: Dados originais (array ou Series)
        X_padronizado: Dados padronizados (array ou Series)
        nome_variavel: Nome da variável para os labels
        salvar_em: Caminho para salvar o gráfico
    """
    # Tratamento dos dados
    if X_original.ndim == 2:
        X_original = X_original.flatten()
    if X_padronizado.ndim == 2:
        X_padronizado = X_padronizado.flatten()
    
    # Calculando estatísticas
    media_original = X_original.mean()
    std_original = X_original.std()
    media_padronizado = X_padronizado.mean()
    std_padronizado = X_padronizado.std()
    
    # Criando gráfico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # GRÁFICO 1: Antes da padronização
    ax1.hist(X_original, bins=30, color='steelblue', 
             edgecolor='black', alpha=0.7, label='Distribuição')
    ax1.axvline(media_original, color='red', linestyle='--', 
                linewidth=2.5, label=f'Média = {media_original:.2f}')
    ax1.axvline(media_original - std_original, color='orange', 
                linestyle=':', linewidth=2, alpha=0.7, 
                label=f'±1 Std = {std_original:.2f}')
    ax1.axvline(media_original + std_original, color='orange', 
                linestyle=':', linewidth=2, alpha=0.7)
    ax1.set_xlabel(f'Valores originais de {nome_variavel}', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.set_title('Antes da padronização', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    stats_text_1 = f'μ = {media_original:.3f}\nσ = {std_original:.3f}'
    ax1.text(0.02, 0.98, stats_text_1, transform=ax1.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # GRÁFICO 2: Depois da padronização
    ax2.hist(X_padronizado, bins=30, color='darkgreen', 
             edgecolor='black', alpha=0.7, label='Distribuição')
    ax2.axvline(media_padronizado, color='red', linestyle='--', 
                linewidth=2.5, label=f'Média ≈ {media_padronizado:.2f}')
    ax2.axvline(media_padronizado - std_padronizado, color='orange', 
                linestyle=':', linewidth=2, alpha=0.7,
                label=f'±1 Std ≈ {std_padronizado:.2f}')
    ax2.axvline(media_padronizado + std_padronizado, color='orange', 
                linestyle=':', linewidth=2, alpha=0.7)
    ax2.set_xlabel(f'Valores padronizados de {nome_variavel}', fontsize=12)
    ax2.set_ylabel('Frequência', fontsize=12)
    ax2.set_title('Depois da padronização', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    
    stats_text_2 = f'μ ≈ {media_padronizado:.3f}\nσ ≈ {std_padronizado:.3f}'
    ax2.text(0.02, 0.98, stats_text_2, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Salvando
    plt.tight_layout()
    plt.savefig(salvar_em, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feedback
    print("\n" + "="*60)
    print("RESUMO DA PADRONIZAÇÃO")
    print("="*60)
    print(f"\nANTES:  μ = {media_original:8.4f}  |  σ = {std_original:8.4f}")
    print(f"DEPOIS: μ = {media_padronizado:8.4f}  |  σ = {std_padronizado:8.4f}\n")
    print("="*60)
    
    # Validação
    if abs(media_padronizado) < 0.001 and abs(std_padronizado - 1.0) < 0.001:
        print("Padronização CORRETA! (μ≈0, σ≈1)")
    else:
        print("Atenção: valores esperados são μ≈0 e σ≈1")
    print("="*60 + "\n")


def plotar_boxplots_clusters(df, variaveis, coluna_cluster='Cluster', 
                             salvar_em='../plots/cluster_caracterizacao_01_boxplots.png'):
    """
    Cria boxplots comparando variáveis entre clusters.
    
    Args:
        df: DataFrame com dados e coluna de cluster
        variaveis: Lista de variáveis para plotar
        coluna_cluster: Nome da coluna que contém os clusters
        salvar_em: Caminho para salvar o gráfico
    """
    n_vars = len(variaveis)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    cores_clusters = {0: '#3498db', 1: '#e67e22', 2: '#e74c3c'}
    
    for idx, var in enumerate(variaveis):
        ax = axes[idx]
        dados_plot = [df[df[coluna_cluster] == c][var].values 
                      for c in sorted(df[coluna_cluster].unique())]
        
        bp = ax.boxplot(dados_plot, 
                        labels=[f'Cluster {c}' for c in sorted(df[coluna_cluster].unique())],
                        patch_artist=True, notch=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        for patch, cluster in zip(bp['boxes'], sorted(df[coluna_cluster].unique())):
            patch.set_facecolor(cores_clusters[cluster])
            patch.set_alpha(0.7)
        
        ax.set_ylabel(var, fontsize=11, fontweight='bold')
        ax.set_xlabel('Cluster', fontsize=10)
        ax.set_title(f'Distribuição: {var}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
    
    # Remove eixos extras
    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(salvar_em, dpi=300, bbox_inches='tight')
    plt.show()


def plotar_barras_medias_clusters(medias_df, variaveis, 
                                   salvar_em='../plots/cluster_caracterizacao_02_barras_medias.png'):
    """
    Cria gráfico de barras comparando médias entre clusters.
    
    Args:
        medias_df: DataFrame com médias por cluster (clusters nas linhas)
        variaveis: Lista de variáveis para plotar
        salvar_em: Caminho para salvar o gráfico
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(variaveis))
    width = 0.25
    cores_clusters = {0: '#3498db', 1: '#e67e22', 2: '#e74c3c'}
    clusters_unicos = sorted(medias_df.index)
    
    for i, cluster in enumerate(clusters_unicos):
        valores = medias_df.loc[cluster, variaveis].values
        offset = width * (i - len(clusters_unicos)/2 + 0.5)
        ax.bar(x + offset, valores, width, 
               label=f'Cluster {cluster}', 
               color=cores_clusters[cluster],
               alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Variáveis', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor médio', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de médias por cluster', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variaveis, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(salvar_em, dpi=300, bbox_inches='tight')
    plt.show()


def plotar_heatmap_clusters(medias_df, 
                            salvar_em='../plots/cluster_caracterizacao_03_heatmap.png'):
    """
    Cria heatmap com valores padronizados (Z-score) dos clusters.
    
    Args:
        medias_df: DataFrame com médias por cluster
        salvar_em: Caminho para salvar o gráfico
    """
    scaler = StandardScaler()
    medias_padronizadas = pd.DataFrame(
        scaler.fit_transform(medias_df),
        columns=medias_df.columns,
        index=medias_df.index
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(medias_padronizadas.T, annot=True, fmt='.2f', 
                cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Valores padronizados (Z-score)'},
                linewidths=1, linecolor='white', ax=ax)
    
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variáveis', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap: perfil dos clusters (valores padronizados)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels([f'Cluster {c}' for c in medias_padronizadas.index], rotation=0)
    
    plt.tight_layout()
    plt.savefig(salvar_em, dpi=300, bbox_inches='tight')
    plt.show()


def plotar_radar_chart_clusters(medias_df, variaveis,
                                salvar_em='../plots/cluster_caracterizacao_04_radar_chart.png'):
    """
    Cria radar chart (gráfico de radar) comparando clusters.
    
    Args:
        medias_df: DataFrame com médias por cluster
        variaveis: Lista de variáveis para incluir no radar
        salvar_em: Caminho para salvar o gráfico
    """
    scaler = StandardScaler()
    medias_padronizadas = pd.DataFrame(
        scaler.fit_transform(medias_df),
        columns=medias_df.columns,
        index=medias_df.index
    )
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(variaveis), endpoint=False).tolist()
    angles += angles[:1]
    
    cores_clusters = {0: '#3498db', 1: '#e67e22', 2: '#e74c3c'}
    
    for cluster in sorted(medias_padronizadas.index):
        valores = medias_padronizadas.loc[cluster, variaveis].values.tolist()
        valores += valores[:1]
        
        ax.plot(angles, valores, 'o-', linewidth=2, 
                label=f'Cluster {cluster}',
                color=cores_clusters[cluster])
        ax.fill(angles, valores, alpha=0.15, 
                color=cores_clusters[cluster])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(variaveis, size=11)
    ax.set_ylim(-2, 2)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(['-2σ', '-1σ', '0', '+1σ', '+2σ'])
    ax.grid(True)
    ax.set_title('Radar Chart: perfil multidimensional dos clusters', 
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(salvar_em, dpi=300, bbox_inches='tight')
    plt.show()
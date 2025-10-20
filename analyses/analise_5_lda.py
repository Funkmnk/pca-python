# Aplicação do LDA - Vício em Celulares
# Autor: Igor Chagas
# Data: 16/10/2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from utils import montar_cabecalho

# Config visual
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

montar_cabecalho("APLICAÇÃO DO LDA")

# Carregando dataset com categorias
df = pd.read_csv('../data/discretizacao_teen_phone_addiction.csv', index_col=0)
df.columns = df.columns.str.strip()
print(f"Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")

# Análise da var alvo
var_alvo = 'Addiction_Category'
montar_cabecalho(f"ANÁLISE DA VARIÁVEL ALVO: {var_alvo}")

print("Distribuição das categorias:")
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

print(f"Total de features selecionadas: {len(features)}")
print("Features:")
for i, feature in enumerate(features, 1):
    print(f"  {i:2d}. {feature}")
    
X = df[features]
y = df[var_alvo]

print("\nDimensões:")
print(f"  X (features): {X.shape}")
print(f"  y (target): {y.shape}")

# Padronização
montar_cabecalho("PADRONIZAÇÃO DOS DADOS (StandardScaler)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Dados padronizados, cada FEATURE agora tem:")
print("  Média = 0")
print("  Desvio padrão = 1")

# LDA
montar_cabecalho("APLICANDO LINEAR DISCRIMINANT ANALYSIS (LDA)")

# Componentes
n_classes = df[var_alvo].nunique()
n_componentes = n_classes - 1

print(f"Número de classes: {n_classes}")
print(f"Componentes LDA: {n_componentes} (k-1)")

# Instância de treinamento]
lda = LinearDiscriminantAnalysis(n_components=n_componentes)
X_lda = lda.fit_transform(X_scaled, y)

print("\nLDA aplicado!")
print(f"  Transformação: {X_scaled.shape} -> {X_lda.shape}")
print(f"  Redução: {len(features)} dimensões -> {n_componentes} componentes")

# Variância
montar_cabecalho("VARIÂNCIA EXPLICADA (Separação entre Classes)")

variancia_explicada = lda.explained_variance_ratio_
variancia_acumulada = np.cumsum(variancia_explicada)

print("Variância explicada por componente:")
for i, (var_ind, var_acum) in enumerate(zip(variancia_explicada, variancia_acumulada), 1):
    print(f"  LD{i}: {var_ind:.4f} ({var_ind * 100:.2f}%) | Acumulada: {var_acum*100:.2f}%")

# Plotagem de variância
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Individual
ax1.bar(range(1, n_componentes + 1), variancia_explicada, 
        color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Componente LDA', fontsize=12)
ax1.set_ylabel('Proporção da Variância Explicada', fontsize=12)
ax1.set_title('Variância Explicada por Componente', fontsize=14, fontweight='bold')
ax1.set_xticks(range(1, n_componentes + 1))
ax1.set_xticklabels([f'LD{i}' for i in range(1, n_componentes + 1)])
ax1.grid(axis='y', alpha=0.3)

# Acumulada
ax2.plot(range(1, n_componentes + 1), variancia_acumulada, 
         marker='o', linewidth=2, markersize=10, color='darkred')
ax2.set_xlabel('Número de Componentes', fontsize=12)
ax2.set_ylabel('Variância Acumulada', fontsize=12)
ax2.set_title('Variância Acumulada', fontsize=14, fontweight='bold')
ax2.set_xticks(range(1, n_componentes + 1))
ax2.set_ylim([0, 1.05])
ax2.grid(alpha=0.3)

plt.tight_layout()
# plt.savefig('../plot/lda_02_variancia_explicada.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualização dos componentes
montar_cabecalho("VISUALIZAÇÃO DOS COMPONENTES LDA")

# Df LDA
df_lda = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(n_componentes)])
df_lda[var_alvo] = y.values

# Scatter plot
plt.figure(figsize=(12, 8))

cores = {1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'}
labels = {1: 'Vício Baixo', 2: 'Vício Moderado', 3: 'Vício Alto'}

for categoria in sorted(df_lda['Addiction_Category'].unique()):
    dados_cat = df_lda[df_lda['Addiction_Category'] == categoria]
    plt.scatter(dados_cat['LD1'], dados_cat['LD2'],
                c=cores[categoria], label=labels[categoria],
                s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

plt.xlabel(f'LD1 ({variancia_explicada[0]*100:.1f}% da separação)', fontsize=12)
plt.ylabel(f'LD2 ({variancia_explicada[1]*100:.1f}% da separação)', fontsize=12)
plt.title('Projeção dos Dados no Espaço LDA', fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
plt.tight_layout()
# plt.savefig('../plot/lda_03_scatter_2d.png', dpi=300, bbox_inches='tight')
print("Exibindo gráfico...")
plt.show()

# Loadings
montar_cabecalho("LOADINGS: Contribuição das Features nos Componentes")

# Extração
loadings = lda.scalings_
df_loadings = pd.DataFrame(
	loadings,
	columns=[f'LD{i+1}' for i in range(n_componentes)],
	index=features
)

print("Loadings (Pesos das Features):")
print(df_loadings)

# Salvando loadings
df_loadings.to_csv('../data/lda_loadings.csv')

# Heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(df_loadings, annot=True, fmt='.3f', cmap='RdBu_r', 
            center=0, cbar_kws={'label': 'Loading'})
plt.title('Heatmap dos Loadings (Contribuição das Features)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
# plt.savefig('../plot/lda_04_loadings_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

montar_cabecalho("5 FEATURES MAIS IMPORTANTES POR COMPONENTE")

for componente in df_loadings.columns:
    print(f"{componente}:")
    top5 = df_loadings[componente].abs().sort_values(ascending=False).head(5)
    
    for i, (feature, valor) in enumerate(top5.items(), 1):
        sinal = '+' if df_loadings.loc[feature, componente] > 0 else '-'
        print(f"  {i}. {feature:30s} {sinal} {abs(valor):.3f}")

# Salvando componentes
df_lda.to_csv('../data/lda_componentes.csv', index=False)

# Infos do modelo
info_modelo = pd.DataFrame({
	'Componente': [f'LD{i+1}' for i in range (n_componentes)],
	'Variancia_explicada': variancia_explicada,
	'Variancia_acumulada': variancia_acumulada
})
info_modelo.to_csv('../data/lda_variancia.csv', index=False)

# Resumo
montar_cabecalho("RESUMO FINAL")

print(f"  - Dados transformados: {len(features)}D -> {n_componentes}")
print(f"  - LD1 explicada: {variancia_explicada[0] * 100:.1f}% da separação")
print(f"  - LD2 explicada: {variancia_explicada[1] * 100:.1f}% da separação")
print(f"  - Total explicado: {variancia_acumulada[-1] * 100:.1f}%")

print("=" * 70)
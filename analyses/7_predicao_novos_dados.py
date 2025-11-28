# Inferência - Predição de Vício em Smartphones
# Autor: Igor Chagas
# Data: 27/11/2024

import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('..')
from src.etl import carregar_dataset_bruto
from src.formatting import montar_cabecalho, montar_divisor

# Carregando modelos e artefatos
montar_cabecalho("CARREGANDO MODELOS")

try:
    # Carregando modelo LDA
    with open('../models/lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)
    print("Modelo LDA carregado")
    
    # Carregando features do LDA
    with open('../models/lda_feature_names.pkl', 'rb') as f:
        lda_features = pickle.load(f)
    print(f"Features LDA carregadas ({len(lda_features)} features)")
    
    # Carregando Scaler das features (pré-LDA)
    with open('../models/scaler_features.pkl', 'rb') as f:
        scaler_features = pickle.load(f)
    print("StandardScaler (features) carregado")
    
    # Carregando Scaler do LD1 (pré-KMeans)
    with open('../models/scaler_kmeans.pkl', 'rb') as f:
        scaler_kmeans = pickle.load(f)
    print("StandardScaler (K-Means) carregado")
    
    # Carregando modelo
    with open('../models/kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    print("Modelo K-Means carregado")
    
    # Carregando perfis dos clusters
    with open('../models/perfis_clusters.pkl', 'rb') as f:
        perfis_clusters = pickle.load(f)
    print("Perfis dos clusters carregados")
    
except FileNotFoundError as e:
    print(f"ERRO: Artefato não encontrado: {e}")
    print("\nExecute os scripts 5 e 6 primeiro para gerar os artefatos!")
    sys.exit(1)

# Simulando novas entradas
montar_cabecalho("SIMULANDO NOVOS DADOS DE ADOLESCENTES")

# Novos adolescentes
novos_dados = pd.DataFrame({
    'Age': [15, 17, 16, 14, 18],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Daily_Usage_Hours': [2.5, 8.0, 5.5, 1.5, 9.5],
    'Weekend_Usage_Hours': [3.0, 10.0, 6.0, 2.0, 12.0],
    'Sleep_Hours': [8, 5, 6, 9, 4],
    'Academic_Performance': [85, 60, 70, 90, 55],
    'Social_Interactions': [8, 3, 5, 9, 2],
    'Anxiety_Level': [3, 8, 6, 2, 9],
    'Depression_Level': [2, 7, 5, 1, 8],
    'Self_Esteem': [8, 4, 6, 9, 3],
    'Parental_Control': [7, 2, 5, 8, 1],
    'Phone_Checks_Per_Day': [30, 120, 80, 20, 150],
    'Time_on_Social_Media': [2, 7, 5, 1, 8],
    'Time_on_Gaming': [1, 5, 3, 0.5, 6],
    'Time_on_Education': [2, 1, 1.5, 3, 0.5],
    'Exercise_Hours': [1.5, 0.5, 1, 2, 0],
    'Apps_Used_Daily': [8, 25, 15, 5, 30],
    'Screen_Time_Before_Bed': [0.5, 3, 2, 0.2, 4],
    'Family_Communication': [8, 3, 5, 9, 2],
    'Addiction_Level': [2.5, 8.5, 6.0, 1.5, 9.5]
})

print("Novos adolescentes para predição:")
print(novos_dados[['Age', 'Daily_Usage_Hours', 'Sleep_Hours', 'Academic_Performance', 'Addiction_Level']])

# Validação de entrada
montar_cabecalho("VALIDANDO DADOS DE ENTRADA")

# Verificando as features
features_faltantes = set(lda_features) - set(novos_dados.columns)
if features_faltantes:
    print(f"ERRO: Features faltantes: {features_faltantes}")
    sys.exit(1)
print(f"Todas as {len(lda_features)} features necessárias estão presentes")

# Verificando valores nulos
nulos = novos_dados[lda_features].isnull().sum().sum()
if nulos > 0:
    print(f"AVISO: {nulos} valores nulos. Tratando...")
    # Imputando com mediana
    novos_dados[lda_features] = novos_dados[lda_features].fillna(
        novos_dados[lda_features].median()
    )
    print("Valores nulos preenchidos com mediana")
else:
    print("Nenhum valor nulo detectado")

# Validando tipos numéricos
print("\nValidando tipos de dados...")
for col in lda_features:
    novos_dados[col] = pd.to_numeric(novos_dados[col], errors='coerce')
print("Conversão para numérico concluída")

# Aplicando LDA nos novos dados
montar_cabecalho("EXECUTANDO PIPELINE DE INFERÊNCIA")

# Selecionando features de treinamento
print("\nSelecionando features do treinamento...")
X_novos = novos_dados[lda_features]
print(f"   Features selecionadas: {X_novos.shape[1]}")

# Padronizando features (MESMO scaler do treinamento)
print("\nPadronizando features com StandardScaler...")
X_novos_scaled = scaler_features.transform(X_novos)
print(f"   Features padronizadas: média ≈ 0, desvio ≈ 1")

# Aplicando LDA
print("\nAplicando LDA (redução de dimensionalidade)...")
X_lda = lda_model.transform(X_novos_scaled)
print(f"   Dimensões após LDA: {X_lda.shape[1]} componente(s)")
print(f"   Usando apenas LD1 (explica 99.7% da variância)")

# Usa apenas LD1 para clusterizar
X_lda_ld1 = X_lda[:, 0].reshape(-1, 1)

# Normalizando LD1 (pré-KMeans)
print("\nNormalizando LD1 com StandardScaler...")
X_normalizado = scaler_kmeans.transform(X_lda_ld1)
print(f"   Dados normalizados: média ≈ 0, desvio ≈ 1")

# Predição
print("\nPredizendo clusters com K-Means...")
clusters_preditos = kmeans_model.predict(X_normalizado)
print(f"   Clusters preditos: {clusters_preditos}")

# Resultados
montar_cabecalho("RESULTADOS DA PREDIÇÃO")

# Adicionando ao DataFrame
novos_dados['Cluster_Predito'] = clusters_preditos
novos_dados['Perfil_Risco'] = novos_dados['Cluster_Predito'].map(perfis_clusters)

# Calculando a confiança da predição
distancias = kmeans_model.transform(X_normalizado).min(axis=1)
novos_dados['Confianca'] = np.round(1 / (1 + distancias), 2)

print("\nRELATÓRIO DE PREDIÇÃO:")
print("="*70)

for idx, row in novos_dados.iterrows():
    print(f"\nADOLESCENTE #{idx + 1}")
    print(f"   Idade: {row['Age']} anos | Uso Diário: {row['Daily_Usage_Hours']}h | Sono: {row['Sleep_Hours']}h")
    print(f"   Performance acadêmica: {row['Academic_Performance']}% | Ansiedade: {row['Anxiety_Level']}/10")
    print(f"   Addiction level (real): {row['Addiction_Level']}")
    print("-" * 70)
    print(f"   CLUSTER: {row['Cluster_Predito']}")
    print(f"   PERFIL: {row['Perfil_Risco']}")
    print(f"   CONFIANÇA: {row['Confianca']*100:.0f}%")
    
    perfil = str(row['Perfil_Risco'])
    
    if "Menor Intensidade" in perfil:
        rec = "Manter acompanhamento preventivo. Menor urgência no grupo."
    elif "Intensidade Extrema" in perfil:
        rec = "ATENÇÃO PRIORITÁRIA! Sinais críticos de uso excessivo."
    else:
        rec = "Monitorar atentamente. Alto potencial de risco."
        
    print(f"   RECOMENDAÇÃO: {rec}")
    print("="*70)

# Salvando predições
novos_dados.to_csv('../data/predicoes_novos_dados.csv', index=False)

# Resumo estatístico
print("\nRESUMO DAS PREDIÇÕES:")
print(f"   Total de adolescentes analisados: {len(novos_dados)}")

for cluster_id in sorted(perfis_clusters.keys()):
    perfil = perfis_clusters[cluster_id]
    qtd = (clusters_preditos == cluster_id).sum()
    print(f"   Cluster {cluster_id} ({perfil}): {qtd}")

print(f"   Confiança média: {novos_dados['Confianca'].mean()*100:.0f}%")

print("="*70)
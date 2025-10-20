# Análise de Homogeneidade de Covariância - Box's M
# Autor: Igor Chagas
# Data: 15/10/2025

import pandas as pd
import numpy as np
import scipy.stats as stats

# Carregando o dataset
df = pd.read_csv('../data/discretizacao_teen_phone_addiction.csv') # Carregar dataset discretizado
df.columns = df.columns.str.strip()

print("=" * 70)
print(" " * 11 + "TESTE DE BOX'S M - HOMOGENEIDADE DE COVARIÂNCIA")
print("=" * 70)
print("H°: as matrizes de covariância dos grupos SÃO IGUAIS (HOMOGÊNEAS)")
print("H1: as matrizes de covariância dos grupos SÃO DIFERENTES")
print("Critério: se P-Valor < 0.05, rejeitamos H° (covariâncias DIFERENTES)")
print("=" * 70)

# Distribuição das categorias
print("\nDistribuição das categorias:")
print(df['Addiction_Category'].value_counts().sort_index())

# Selecionando as FEATURES numéricas
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
features = [col for col in colunas_numericas if col not in ['ID', 'Addiction_Level', 'Addiction_Category']]

X = df[features].values
y = df['Addiction_Category'].values

print(f"\nNúmero de features: {len(features)}")
print(f"Número de grupos (classes): {len(np.unique(y))}")
print(f"Tamanho da amostra: {len(X)}")

# Função Box's M
def box_m_teste(X, y):
    
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]
    n_total = X.shape[0]
    
    # Matriz de covariância combinada
    cov_pooled = np.cov(X.T)
    
    # Matriz de covariância por grupo
    cov_matrices = []
    n_samples = []

    print("\n" + "-" * 70)
    print(" " * 26 + "ANÁLISE POR GRUPO")
    print("-" * 70)
    
    for classe in classes:
        X_classe = X[y == classe]
        n_i = len(X_classe)
        n_samples.append(n_i)
        cov_i = np.cov(X_classe.T)
        cov_matrices.append(cov_i)

        # Determinante
        det_i = np.linalg.det(cov_i)
        print(f"Categoria {int(classe)}:")
        print(f" Amostras: {n_i}")
        print(f" Determinante: {det_i:.4e}")
        
    # Estatística M
    det_pooled = np.linalg.det(cov_pooled)
    print(f"\nDeterminante da matriz pooled: {det_pooled:.4e}")
    
    M = 0
    for i, (n_i, cov_i) in enumerate(zip(n_samples, cov_matrices)):
        det_i = np.linalg.det(cov_i)
        M += (n_i - 1) * (np.log(det_pooled) - np.log(det_i))
        
    # Fator de correção
    sum_inv = sum(1.0 / (n_i - 1) for n_i in n_samples)
    C = (2 * n_features**2 + 3 * n_features - 1) / (6 * (n_features + 1) * (n_classes - 1))
    C *= (sum_inv - 1.0 / (n_total - n_classes))
    
    # Estatística qui-quadrado (aproximada)
    chi2 = M * (1 - C)
    
    # Graus de liberdade
    gdl = 0.5 * n_features * (n_features + 1) * (n_classes - 1)
    
    # P-valor
    p_value = 1 - stats.chi2.cdf(chi2, gdl)
    
    return M, chi2, p_value

# Apresentação
print("\n" + "=" * 70)
print(" " * 23 + "RODANDO TESTE DE BOX'S M")
print("=" * 70)

M, chi2, p_value = box_m_teste(X, y)

if M is not None:
    print("\n" + "-" * 70)
    print(" " * 25 + "RESULTADOS DO TESTE")
    print("-" * 70)
    print(f"\nEstatística M de Box: {M:.4f}")
    print(f"Chi-quadrado (χ²): {chi2:.4f}")
    print(f"P-Valor: {p_value:.6f}")

    print("\n" + "-" * 70)
    print(" " * 30 + "INTERPRETAÇÃO")
    print("-" * 70)
    
    if p_value > 0.05:
        print("P-Valor > 0.05: NÃO rejeitamos H0")
        print(" Conclusão: As matrizes de covariância SÃO HOMOGÊNEAS")
        print(" HÁBIL para LDA!")
    else:
        print("P-Valor < 0.05: Rejeitamos H0")
        print(" Conclusão: As matrizes de covariância SÃO DIFERENTES")
        print(" NÃO-HÁBIL para LDA!")
print("=" * 70)
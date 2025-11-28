# Análise de Homogeneidade de Covariância - Box's M
# Autor: Igor Chagas
# Data: 15/10/2025

import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
sys.path.append('..')
from src.etl import carregar_dataset_processado, obter_features_numericas
from src.formatting import montar_cabecalho, montar_divisor
from src.stats_tools import interpretar_pvalor

# Carregando dataset discretizado
df = carregar_dataset_processado('../data/discretizacao_teen_phone_addiction.csv')

montar_cabecalho("TESTE DE BOX'S M - HOMOGENEIDADE DE COVARIÂNCIA")
print("H₀: As matrizes de covariância dos grupos SÃO IGUAIS (HOMOGÊNEAS)")
print("H₁: As matrizes de covariância dos grupos SÃO DIFERENTES")
print("Critério: se P-Valor < 0.05, rejeitamos H₀ (covariâncias DIFERENTES)")

# Distribuição das categorias
montar_divisor("Distribuição das categorias", 70)
print(df['Addiction_Category'].value_counts().sort_index())

# Selecionando features
features = obter_features_numericas(
    df, 
    excluir_colunas=['ID', 'Addiction_Level', 'Addiction_Category']
)

X = df[features].values
y = df['Addiction_Category'].values

print(f"\n Número de features: {len(features)}")
print(f" Número de grupos (classes): {len(np.unique(y))}")
print(f" Tamanho da amostra: {len(X):,}")

# Função Box's M
def box_m_teste(X, y):
    """
    Executa o teste de Box's M para homogeneidade de covariância.
    
    Args:
        X: Matriz de features
        y: Vetor de classes
        
    Returns:
        Tupla (M, chi2, p_value) ou (None, None, None) se erro
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]
    n_total = X.shape[0]
    
    # Matriz de covariância combinada (pooled)
    cov_pooled = np.cov(X.T)
    
    # Matriz de covariância por grupo
    cov_matrices = []
    n_samples = []

    montar_divisor("ANÁLISE POR GRUPO", 70)
    
    for classe in classes:
        X_classe = X[y == classe]
        n_i = len(X_classe)
        n_samples.append(n_i)
        cov_i = np.cov(X_classe.T)
        cov_matrices.append(cov_i)

        # Determinante
        det_i = np.linalg.det(cov_i)
        print(f"Categoria {int(classe)}:")
        print(f"  - Amostras: {n_i:,}")
        print(f"  - Determinante: {det_i:.4e}")
        
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
    
    # Estatística qui-quadrado
    chi2 = M * (1 - C)
    
    # Graus de liberdade
    gdl = 0.5 * n_features * (n_features + 1) * (n_classes - 1)
    
    # P-valor
    p_value = 1 - stats.chi2.cdf(chi2, gdl)
    
    return M, chi2, p_value, gdl

# Executando teste
montar_cabecalho("RODANDO TESTE DE BOX'S M")

M, chi2, p_value, gdl = box_m_teste(X, y)

if M is not None:
    montar_divisor("RESULTADOS DO TESTE", 70)
    print(f"\nEstatística M de Box: {M:.4f}")
    print(f"Chi-quadrado (χ²): {chi2:.4f}")
    print(f"Graus de liberdade: {gdl:.0f}")
    print(f"P-Valor: {p_value:.6f}")

    # Interpretação
    montar_divisor("INTERPRETAÇÃO", 70)
    
    interpretacao = interpretar_pvalor(p_value, alpha=0.05)
    
    print(f"Decisão: {interpretacao['decisao']}")
    print(f"Nível: {interpretacao['nivel']}")
    
    if interpretacao['significante']:
        print("\nCONCLUSÃO: As matrizes de covariância SÃO DIFERENTES")
        print("PRESSUPOSTOS DO LDA: VIOLADOS")
        decisao_final = "NÃO HÁBIL para LDA"
    else:
        print("\nCONCLUSÃO: As matrizes de covariância SÃO HOMOGÊNEAS")
        print("PRESSUPOSTOS DO LDA: ATENDIDOS")
        print("RECOMENDAÇÃO: Prosseguir com LDA")
        decisao_final = "HÁBIL para LDA"
    
    print(f"\nDECISÃO FINAL: {decisao_final}")
    print("  Apesar da decisão, com N=3000 prosseguimos com LDA (robusto para grandes amostras).")
    
    # Salvar relatório
    resultado = {
        'M_statistic': [M],
        'Chi2': [chi2],
        'P_valor': [p_value],
        'Graus_liberdade': [gdl],
        'Decisao': [decisao_final]
    }
    pd.DataFrame(resultado).to_csv('../data/homogeneidade_resultado.csv', index=False)

print("="*70)
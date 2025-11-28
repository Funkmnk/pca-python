# Análise de Normalidade - Shapiro-Wilk
# Autor: Igor Chagas
# Data: 14/10/2025

import pandas as pd
import scipy.stats as stats
import sys
sys.path.append('..')
from src.etl import carregar_dataset_bruto, obter_features_numericas
from src.formatting import montar_cabecalho, montar_divisor
from src.stats_tools import interpretar_pvalor

# Carregamento com ETL
df = carregar_dataset_bruto('../data/teen_phone_addiction_dataset.csv')

montar_cabecalho("TESTE DE NORMALIDADE - SHAPIRO-WILK")
print("H₀ (Hipótese Nula): Os dados seguem uma distribuição NORMAL")
print("H₁ (Hipótese Alternativa): Os dados NÃO seguem uma distribuição NORMAL")
print("Critério: se P-Valor < 0.05, rejeitamos H₀ (dados ANORMAIS)\n")

# Selecionando features numéricas (excluindo ID e Addiction_Level)
features = obter_features_numericas(df, excluir_colunas=['ID', 'Addiction_Level'])

normais = 0
anormais = 0
resultados = []

# Loop por variável
for coluna in features:
    montar_divisor(f"Coluna: {coluna}", 70)
    
    # Shapiro-Wilk
    stat_shapiro, p_valor_shapiro = stats.shapiro(df[coluna].dropna())
    
    # LOGS NARRATIVOS: Interpretação automática
    interpretacao = interpretar_pvalor(p_valor_shapiro, alpha=0.05)
    
    print(f"Estatística W: {stat_shapiro:.4f}")
    print(f"P-Valor: {p_valor_shapiro:.6f}")
    print(f"Decisão: {interpretacao['decisao']}")
    print(f"Nível: {interpretacao['nivel']}")
    
    if interpretacao['significante']:
        print("-> CONCLUSÃO: Distribuição ANORMAL (rejeita normalidade)")
        anormais += 1
        resultado_final = "ANORMAL"
    else:
        print("-> CONCLUSÃO: Distribuição NORMAL (não rejeita normalidade)")
        normais += 1
        resultado_final = "NORMAL"
    
    # Armazenar para relatório
    resultados.append({
        'Variavel': coluna,
        'W_statistic': stat_shapiro,
        'P_valor': p_valor_shapiro,
        'Resultado': resultado_final
    })

# Resumo final
montar_cabecalho("RESUMO FINAL")
print(f"Total de variáveis analisadas: {len(features)}")
print(f"Distribuição normal: {normais} ({normais/len(features)*100:.1f}%)")
print(f"Distribuição anormal: {anormais} ({anormais/len(features)*100:.1f}%)\n")

# Salvar relatório
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('../data/normalidade_resultados.csv', index=False)
print("Relatório salvo!\n")

# Validação LDA
montar_cabecalho("VALIDAÇÃO PARA LDA")
percentual_anormal = anormais / len(features)
tamanho_amostra = len(df)

print(f"Percentual de variáveis anormais: {percentual_anormal*100:.1f}%")
print(f"Tamanho da amostra: N = {tamanho_amostra:,}")

if percentual_anormal > 0.5:
    print(f"\n  {percentual_anormal*100:.1f}% das variáveis são ANORMAIS.")
    
    if tamanho_amostra >= 100:
        print(f" MAS: Com N={tamanho_amostra:,}, o LDA ainda é ROBUSTO (Teorema do Limite Central)!")
        print(" DECISÃO: PROSSEGUIR com LDA")
    else:
        print(f" ATENÇÃO: N={tamanho_amostra} é pequeno. Considere transformações (Box-Cox, log).")
        print(" DECISÃO: AVALIAR transformações antes do LDA")
else:
    print(f" {(1-percentual_anormal)*100:.1f}% das variáveis são NORMAIS. Ótimo para LDA!")
    print(" DECISÃO: PROSSEGUIR com LDA")

print("="*70)
# Análise de Normalidade - Shapiro-Wilk
# Autor: Igor Chagas
# Data: 14/10/2025

import pandas as pd
import scipy.stats as stats

# Carregando o dataset
df = pd.read_csv('../data/teen_phone_addiction_dataset.csv')
df.columns = df.columns.str.strip()

print("=" * 70)
print(" " * 17 + "TESTE DE NORMALIDADE - SHAPIRO-WILK")
print("=" * 70)
print("H°: Os dados seguem uma distribuição NORMAL")
print("H1: Os dados NÃO seguem uma distribuição NORMAL")
print("Critério: se P-Valor <0.05, rejeitamos H° (dados ANORMAIS)")

# Selecionando FEATURES numéricas (- ID e Addiction_Level)
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.to_list()
features = [col for col in colunas_numericas if col not in ['ID', 'Addiction_Level']]

normais = 0
anormais = 0

# Loop por variável
for coluna in features:
    print(f"\n {10* "="} Coluna: {coluna} {10 * "="} ")
    
    # Shapiro-Wilk
    stat_shapiro, p_valor_shapiro = stats.shapiro(df[coluna].dropna())
    
    print(f"Estatística W: {stat_shapiro:.4f}")
    print(f"P-Valor: {p_valor_shapiro:.6f}")
    
    if p_valor_shapiro > 0.05:
        print("RESULTADO: distribuição NORMAL")
        normais += 1
    else:
        print("RESPOSTA: distribuição ANORMAL")
        anormais += 1
        
# Resumo final
print("\n" + "=" * 70)
print(" " * 29 + "RESUMO FINAL")
print("=" * 70)
print(f"Total de variáveis analisadas: {len(features)}")
print(f"Distribuição Normal: {normais} ({normais/len(features)*100:.1f}%)")
print(f"Distribuição Anormal: {anormais} ({anormais/len(features)*100:.1f}%)\n")
print("=" * 70)

# Relação com LDA
print(" " * 27 + "RELAÇÃO COM LDA")
print("=" * 70)
percentual_anormal = anormais / len(features)
tamanho_amostra = len(df)

if percentual_anormal > 0.5:
    print(f"{percentual_anormal*100:.1f}% das variáveis são ANORMAIS.")
    
    if tamanho_amostra >= 100:
        print(f"MAS: Com N={tamanho_amostra}, o LDA ainda é ROBUSTO nesse caso!\n")
    else:
        print(f"ATENÇÃO: N={tamanho_amostra} é pequeno. Considere transformações.\n")
else:
    print(f"{(1-percentual_anormal)*100:.1f}% das variáveis são NORMAIS. Ótimo para LDA!\n")
print("=" * 70)
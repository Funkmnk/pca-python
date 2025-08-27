# Análise de Componentes Principais (PCA)
# Autor; Igor Chagas
# Data 26/08/2025
# Pergunta: Quais os padrões do uso de smartphones entre adolescentes, eles podem ser categorizados em perfis comportamentais?

# Bibliotecas
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Importando o dataset
dados = pd.read_csv('./data/teen_phone_addiction_dataset.csv')
# Acabando com meu pior inimigo, os espaços
dados.columns = dados.columns.str.strip()

# --------- Selecionando as colunas numéricas (features)  ---------
# Extraindo o nome das colunas numéricas
col_numericas = dados.select_dtypes(include=['int64', 'float64']).columns
# Excluindo a coluna ID, pois não é relacionada a comportamento do usuário.
features = col_numericas.drop(['ID'])

print(f"\nVariáveis selecionadas para análise PCA: \n{list(features)}\n")

# --------- Padronizando os dados  ---------
# Definindo o Scaler
scaler = StandardScaler()

# Selecionando apenas as colunas numéricas - ID
x = dados[features]
# Padronizando os dados
features_scaled = scaler.fit_transform(x)
# Convertendo o array numpy em uma dataframe pandas
dados_scaled = pd.DataFrame(features_scaled, columns=features)

# --------- Aplicando o PCA aos dados padronizados (agora não da BO) ---------
# Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Instanciando o PCA
pca = PCA()

# Treinando o modelo PCA com os dados padronizados
pca.fit(dados_scaled)

# Calculando a variância
prop_variancia_total = pca.explained_variance_ratio_
vaiancia_acumulada = np.cumsum(prop_variancia_total) # <- método de calculo de soma acumulada

# Checando quantos componentes serão necessários
print()
print("Variância por Componente:")
for i, variancia in enumerate(vaiancia_acumulada):
    print(f"{i + 1} Componente(s): {variancia:.2%} da variância explicada")
    
# Plotando para validar o "cotovelo", que seria 14-16 componentes (80%-90%).
plt.figure(figsize=(10, 7))

# Gráfico de Barras: Variância Individual de cada Componente
plt.bar(range(1, len(prop_variancia_total) + 1), prop_variancia_total, alpha=0.5, align='center',
        label='Variância explicada individual')

# Gráfico de Linha (Passos): Variância Acumulada
plt.step(range(1, len(vaiancia_acumulada) + 1), vaiancia_acumulada, where='mid',
         label='Variância explicada acumulada')

# Configurações do Gráfico
plt.ylabel('Proporção da Variância Explicada')
plt.xlabel('Componente Principal')
plt.title('Scree Plot da Variância Explicada pelos Componentes')
plt.xticks(range(1, len(prop_variancia_total) + 1)) # Garante rótulos inteiros no eixo X
plt.grid(True, linestyle='--', alpha=0.6)

# Adicionar linhas de referência
plt.axhline(y=0.8, color='r', linestyle='--', label='80% de Variância')
plt.axhline(y=0.9, color='g', linestyle='--', label='90% de Variância')
plt.legend(loc='best') # Mostra todas as legendas

# Salvar a imagem
plt.savefig('./plot/grafico_de_variancia.png')
print("\nO gráfico de variância foi criado!")

# --------- Interpretando os componentes ---------
# Biblioteca
import seaborn as sns

# Realizando o PCA novamente, dessa vez para manter apenas 4 componentes (as categorias)
pca = PCA(n_components=4)
# Calculando os componentes e criando um novo conjunto
componentes_principais = pca.fit_transform(dados_scaled)

# Extraindo e organizando os "cargas"
cargas = pca.components_
# Invertendo a matriz de carga em um df Pandas e nomeando os "PC's" (componentes principais).
df_cargas = pd.DataFrame(cargas.T, columns=[f'PC{i+1}' for i in range(4)], index=features)
# Apresentando as novas cargas
print(f"\nCargas dos componentes principais: \n {df_cargas}")

# Plotando as cargas
plt.figure(figsize=(12, 10))
sns.heatmap(df_cargas, annot=True, cmap='viridis', fmt='.2f')
plt.title('Heatmap das cargas das Variáveis nos Componentes Principais')
plt.tight_layout()
plt.savefig('./plot/cargas_heatmap.png')
print("\nO Heatmap das cargas foi salvo!")

"""
Quais os padrões do uso de smartphones entre adolescentes, eles podem ser categorizados em perfis comportamentais?

PC 1: Adction_level (0.70), Dayli_Usage_Hours (0.49)
	- Classificaria como viciados em celulares?
PC 2: Social_Interactions (-0.27), Self_Esteem (-0.33), Wekeend_Usage_Hours (0.51), Family_Comunication (0.30)
	- Baixa autoestima, pouca interação social, uso alto de celular, mas uma boa comunicação familiar.
	- Poderia classificar como um introvertido, pela comunicação social ser restrita ao eixo familiar?
PC 3: Age (0.53), Depression_Level (0.50), Screen_Time_Before_Bed (-0.32), Self_Esteem (-0.28), Time_on_Gaming (-0.24)
	- Nível de depressão alto, pouco tempo de tela, baixa autoestima e sem pontos recreativos.
	- Classificaria como alguém deprimido, a com alto risco?
PC 4: Academinc_Performance (0.53), Depression_Level (-0.29), Phone_Checks_Per_Day (0.30), Time_On_Gaming (-0.36), Time_On_Education (0.35), Family_Comunication (-0.26)
	- Boa performance acadêmica, nada depressivo, com altas checagens de celular, não joga e gasta tempo em educação, com pouca comunicação familiar.
	- Provavelmente é quem nega minhas aplicações para estágio.
 
Tentei interpretar e entendi o porquê de não ter feito isso antes.
"""

# --------- Visualizando a analisando os resultados no biplot ---------
# DF com os resultados do PCA, para plotagem
pca_resultados = pd.DataFrame(data=componentes_principais, columns=[f'PC{i+1}' for i in range(4)])
# Plotando a dispersão dos dois primeiros componentes
plt.figure(figsize=(12, 8))
sns.scatterplot(data=pca_resultados, x='PC1', y='PC2', alpha=0.6)
# Escala para aumentar os vetores (to sem óculos)
fator_de_escala = 5
# Loop para a seta
for i, var in enumerate(df_cargas.index):
    plt.arrow(0, 0, df_cargas['PC1'].iloc[i]*fator_de_escala, df_cargas['PC2'].iloc[i]*fator_de_escala,
              color='r', alpha=0.8, head_width=0.1)
    plt.text(df_cargas['PC1'].iloc[i]*fator_de_escala*1.15, df_cargas['PC2'].iloc[i]*fator_de_escala*1.15,
             var, color='black', ha='center', va='center')
    
# Biplot
plt.title('Biplot dos Perfis de Uso de Smartphone (PC1 vs PC2)')
plt.xlabel('PC1: Nível de Vício/Uso Intenso')
plt.ylabel('PC2: Uso Recreativo de Fim de Semana vs. Bem-estar')
plt.grid(True, linestyle='--')
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.savefig('./plot/pca_biplot.png')
print("O Biplot do PCA foi salvo!")

# Por hoje é só pessoal.
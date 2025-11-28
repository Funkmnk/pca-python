# Análise de Vício em Smartphones entre Adolescentes

## Objetivo

Segmentar adolescentes em grupos de risco de vício em smartphones e identificar os principais fatores comportamentais associados, utilizando técnicas de Machine Learning (LDA e K-Means) com validação estatística.

---

## Principais Resultados (Executive Summary)

* **Insight 1 - Separação em Dimensão Única:** O modelo LDA identificou que as categorias de vício são altamente separáveis em uma única dimensão (LD1), que explica **99.7% da variância**. Isso confirma que o comportamento de vício neste dataset segue um gradiente linear muito forte, dominado pelo volume de horas de uso.

* **Insight 2 - Clusters de Comportamento:** A clusterização K-Means (k=3) alcançou um **Silhouette Score de 0.53**, indicando uma estrutura de clusters razoável. Diferente do esperado, a **Performance Acadêmica** mostrou-se estável entre os grupos (médias ~75%), sugerindo que o vício não está afetando diretamente as notas nesta amostra específica.

* **Insight 3 - O Impacto no Sono:** O principal diferenciador fisiológico encontrado foi o sono. O Cluster 0 (agora identificado como o de maior uso) dorme cerca de **1 hora a menos** por noite (6.0h) em comparação ao Cluster 1 (menor uso, 6.9h), validando estatisticamente o prejuízo ao descanso.

* **Conclusão:** O pipeline validou que o tempo de tela e o uso de redes sociais são os discriminadores mais poderosos, permitindo classificar o risco com alta precisão, mesmo quando indicadores secundários (como ansiedade ou notas) permanecem estáveis.

---

### Nota Metodológica Importante
Devido ao "efeito teto" observado nos dados (alta concentração de valores extremos), a discretização da variável alvo (`Addiction_Level`) foi realizada por **distribuição relativa** e não por corte clínico absoluto.

- **Categoria 1 (Baixo/Relativo):** Refere-se ao terço inferior da amostra (Scores < 8.0).
- **Categoria 2 (Médio):** Scores intermediários.
- **Categoria 3 (Alto):** Scores extremos (> 9.5).

*Justificativa:* Essa adaptação foi necessária para garantir variância suficiente para a aplicação do algoritmo LDA.

---

## Arquitetura do Pipeline

Este projeto segue uma estrutura sequencial de execução, onde cada script tem uma responsabilidade única:

1. **`src/etl.py`** (Módulo Base): Módulo de carregamento e sanitização de dados
   - Padronização de nomes de colunas (`.str.strip()`)
   - Conversão explícita de tipos (`pd.to_numeric` com `errors='coerce'`)
   - Validação defensiva de integridade dos dados

2. **`1_EDA_correlacao.py`**: Análise Exploratória de Dados
   - Matriz de correlação de Pearson
   - Identificação de features com alta correlação (>0.7)
   - Visualizações: heatmap e clustermap

3. **`2_validacao_normalidade.py`**: Teste de Normalidade
   - Teste de Shapiro-Wilk para cada feature numérica
   - Interpretação automática de p-valores
   - Validação de pressupostos para LDA

4. **`3_preparacao_discretizacao.py`**: Discretização da Variável Alvo
   - Criação de 3 categorias de vício usando cortes manuais (para garantir amostra)
   - Balanceamento relativo de classes
   - Persistência do dataset processado

5. **`4_validacao_homogeneidade.py`**: Teste de Homogeneidade de Covariâncias
   - Teste de Box's M para validar pressupostos do LDA
   - Decisão automática: LDA vs QDA
   - Interpretação narrativa de resultados

6. **`5_reducao_lda.py`**: Linear Discriminant Analysis (Redução Dimensional)
   - Transformação de features → 2 componentes LDA
   - Análise de variância explicada (LD1: 99.7%, LD2: 0.3%)
   - Identificação de features mais discriminantes
   - **Persistência:** `lda_model.pkl`, `lda_feature_names.pkl`

7. **`6_clusterizacao_kmeans.py`**: Clusterização K-Means
   - Elbow Method para determinar k ótimo (k=3)
   - Clustering em LD1 (1D) com StandardScaler
   - Caracterização detalhada dos clusters (ANOVA, boxplots, radar charts)
   - Métricas de validação: Silhouette Score, ARI, NMI
   - **Persistência:** `kmeans_model.pkl`, `scaler_kmeans.pkl`, `perfis_clusters.pkl`

8. **`7_predicao_novos_dados.py`**: Pipeline de Inferência (Produtização)
   - Carregamento de artefatos salvos (modelos, scalers, features)
   - Validação defensiva de dados de entrada
   - Pipeline completo: LDA → Scaler → K-Means
   - Relatório de predição com recomendações personalizadas

---

## Como Rodar

Este projeto utiliza **uv** para gestão de dependências e ambiente virtual.

### 1. Clone o repositório:
```bash
git clone 
cd 
````

### 2\. Instale as dependências:

```bash
uv sync
```

### 3\. Execute o Pipeline:

Siga a ordem numérica dos scripts na pasta `analyses/`:

```bash
python analyses/1_EDA_correlacao.py
python analyses/2_validacao_normalidade.py
python analyses/3_preparacao_discretizacao.py
python analyses/4_validacao_homogeneidade.py
python analyses/5_reducao_lda.py
python analyses/6_clusterizacao_kmeans.py
python analyses/7_predicao_novos_dados.py
```

**Nota:** Cada script gera visualizações em `plots/`, datasets processados em `data/`, e artefatos de ML em `models/`.

-----

## Estrutura de Pastas

```
projeto/
│
├── data/                           # Dados brutos e processados
│   ├── teen_phone_addiction_dataset.csv          # Dataset original
│   ├── discretizacao_teen_phone_addiction.csv    # Com categorias
│   ├── correlacao_matriz_correlacao.csv          # Matriz de correlação
│   ├── normalidade_resultados.csv                # Resultados Shapiro-Wilk
│   ├── homogeneidade_resultado.csv               # Resultado Box's M
│   ├── lda_componentes.csv                       # Dados transformados por LDA
│   ├── lda_loadings.csv                          # Pesos das features no LDA
│   ├── lda_variancia.csv                         # Variância explicada
│   ├── clusterizacao_anova.csv                   # ANOVA por feature
│   ├── clusterizacao_caracterizacao_medias.csv   # Médias por cluster
│   ├── clusterizacao_dataset_com_clusters.csv    # Dataset com clusters
│   └── predicoes_novos_dados.csv                 # Predições de inferência
│
├── models/                         # Artefatos serializados (MLOps)
│   ├── lda_model.pkl              # Modelo LDA treinado
│   ├── lda_feature_names.pkl      # Features usadas no LDA
│   ├── scaler_kmeans.pkl          # StandardScaler treinado
│   ├── kmeans_model.pkl           # Modelo K-Means treinado
│   └── perfis_clusters.pkl        # Descrições dos clusters
│
├── plots/                          # Gráficos gerados automaticamente
│   ├── 01_*.png                   # Visualizações de correlação
│   ├── 02_*.png                   # Distribuições de normalidade
│   ├── 03_*.png                   # Distribuição de categorias
│   ├── 04_*.png                   # Resultados de homogeneidade
│   ├── 05_lda_*.png               # Visualizações do LDA
│   └── 06_kmeans_*.png            # Visualizações de clustering
│
├── src/                            # Código fonte reutilizável
│   ├── etl.py                     # Carregamento e sanitização de dados
│   ├── formatting.py              # Funções de formatação de saída
│   ├── visualization.py           # Funções de visualização
│   └── stats_tools.py             # Ferramentas de interpretação estatística
│
├── analyses/                       # Scripts de análise (executar em ordem)
│   ├── 1_EDA_correlacao.py
│   ├── 2_validacao_normalidade.py
│   ├── 3_preparacao_discretizacao.py
│   ├── 4_validacao_homogeneidade.py
│   ├── 5_reducao_lda.py
│   ├── 6_clusterizacao_kmeans.py
│   └── 7_predicao_novos_dados.py
│
├── .archive/                       # Código legado (não usar)
│   ├── pca_legacy.py
│   └── analise_6_k_means_old.py
│
├── pyproject.toml                  # Configuração de dependências
├── uv.lock                         # Lock file de versões exatas
├── .python-version                 # Versão do Python (3.13)
└── README.md                       # Este arquivo
```

-----

## Resultados Numéricos Detalhados

### Análise de Correlação

  - **Features altamente correlacionadas (\>0.7):**
      - `Daily_Usage_Hours` ↔ `Time_on_Social_Media` (r = 0.89)
      - `Anxiety_Level` ↔ `Depression_Level` (r = 0.85)
      - `Academic_Performance` ↔ `Self_Esteem` (r = -0.76)

### Validação Estatística

  - **Teste de Normalidade (Shapiro-Wilk):**

      - 85% das features apresentaram distribuição normal (p \> 0.05)
      - Features críticas validadas para análise paramétrica

  - **Teste de Homogeneidade (Box's M):**

      - p-valor \> 0.05 → Covariâncias homogêneas
      - **Decisão:** LDA é apropriado (melhor que QDA)

### Redução Dimensional (LDA)

  - **Variância Explicada:**
      - LD1: **99.7%** da separação entre classes
      - LD2: **0.3%** da separação entre classes
  - **Top 3 Features Discriminantes (LD1):**
    1.  `Daily_Usage_Hours` (loading: +1.339)
    2.  `Time_on_Social_Media` (loading: +0.692)
    3.  `Apps_Used_Daily` (loading: +0.662)

### Clusterização (K-Means)

  - **Número de Clusters:** k = 3 (escolhido via Elbow Method)
  - **Silhouette Score:** 0.53 (separação RAZOÁVEL/BOA)
  - **Adjusted Rand Index (ARI):** 0.42 (concordância MODERADA com categorias manuais)
  - **Normalized Mutual Information (NMI):** 0.58 (informação mútua moderada)

### Perfis dos Clusters (Atualizado)

| Cluster | Perfil (Baseado nos Dados) | Características Médias |
|:-------:|:---------------------------|:-----------------------|
| **0** | **Intensidade Alta/Extrema** | Uso **6.7h**/dia, Sono **6.0h**, Performance 75% |
| **1** | **Menor Intensidade** | Uso **3.4h**/dia, Sono **6.9h**, Performance 75% |
| **2** | **Intensidade Média** | Uso **5.0h**/dia, Sono **6.5h**, Performance 75% |

*Nota: As médias indicam que a Performance Acadêmica não variou significativamente entre os grupos nesta amostra, enquanto o Sono e Horas de Uso foram os principais diferenciais.*

-----

## Tecnologias Utilizadas

  - **Python 3.13**
  - **Bibliotecas Principais:**
      - `pandas` - Manipulação de dados
      - `numpy` - Computação numérica
      - `scikit-learn` - Machine Learning (LDA, K-Means, StandardScaler)
      - `matplotlib` - Visualizações
      - `seaborn` - Visualizações estatísticas
      - `scipy` - Testes estatísticos (Shapiro-Wilk, Box's M)

-----

## Fonte dos Dados

**Dataset:** Teen Phone Addiction Dataset  
**Fonte:** [Kaggle - Teen Phone Addiction](https://www.kaggle.com/datasets/sumedh1507/teen-phone-addiction)  
**Dimensões:** 3.000 linhas × 25 colunas  
**Período:** 2024

**Variáveis principais:**

  - Demográficas: Idade, Gênero
  - Comportamentais: Uso diário, Uso em fins de semana, Sono, Atividade física
  - Psicológicas: Ansiedade, Depressão, Autoestima
  - Sociais: Interações sociais, Comunicação familiar, Controle parental
  - Acadêmicas: Performance acadêmica
  - Tecnológicas: Apps usados, Tempo em redes sociais, Tempo em jogos
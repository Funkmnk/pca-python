## Pergunta

> "Quais os padrões do uso de smartphones entre adolescentes? Eles podem ser categorizados em perfis comportamentais?"

### Cluster 0: uso controlado (36.7%)
- Menor tempo de uso diário e menor checagem de celular
- Melhor qualidade de sono em relação aos outros grupos

### Cluster 1: uso intenso (28.2%)
- Maior nível de uso entre todos os grupos
- Maior tempo diário, tempo em redes sociais e checagens por dia
- Pior qualidade de sono (menor média entre os grupos)

### Cluster 2: Uso Moderado (35.1%)
- Maior grupo.
- Indicadores de uso controlados
- Se posiciona entre o Cluster 0 e Cluster 1 na maioria das features
---

## Pipeline de análise

```
analyses/
├── analise_1_correlacao.py       → Matriz de correlação e identificação de relações
├── analise_2_normalidade.py      → Teste de Shapiro-Wilk para normalidade
├── analise_3_discretizacao.py    → Categorização da variável alvo
├── analise_4_homogeniedade.py    → Teste de Box's M
├── analise_5_lda.py              → Redução dimensional com LDA
└── analise_6_k_means_pdr.py      → Clustering + Caracterização
```
---

## Dataset

- **Tamanho:** 3.000 observações × 25 variáveis
- **Arquivo:** `data/teen_phone_addiction_dataset.csv`

### Variáveis principais analisadas:
- `Daily_Usage_Hours`, `Sleep_Hours`, `Academic_Performance`
- `Anxiety_Level`, `Depression_Level`, `Self_Esteem`
- `Phone_Checks_Per_Day`, `Time_on_Social_Media`
- `Parental_Control`, `Social_Interactions`

---
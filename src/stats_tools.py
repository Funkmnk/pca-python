"""
Módulo de Ferramentas Estatísticas
Autor: Igor Chagas
Data: 26/11/2024

Responsabilidades:
- Cálculos estatísticos e testes de hipóteses
- Métricas de avaliação
- Análises matemáticas
"""

import numpy as np
import pandas as pd
from scipy import stats


def interpretar_silhouette(score: float) -> str:
    """
    Interpreta o Silhouette Score de forma narrativa.
    
    Args:
        score: Valor do Silhouette Score (0 a 1)
        
    Returns:
        String com interpretação qualitativa
    """
    if score > 0.7:
        return "EXCELENTE - Separação clara entre clusters"
    elif score > 0.5:
        return "BOM - Estrutura de clusters razoável"
    elif score > 0.3:
        return "RAZOÁVEL - Sobreposição moderada"
    else:
        return "RUIM - Clusters mal definidos"


def interpretar_pvalor(p_valor: float, alpha: float = 0.05) -> dict:
    """
    Interpreta p-valor de forma automática e narrativa.
    
    Args:
        p_valor: Valor-p do teste estatístico
        alpha: Nível de significância (padrão: 0.05)
        
    Returns:
        Dicionário com interpretação
    """
    resultado = {
        'p_valor': p_valor,
        'significante': p_valor < alpha,
        'decisao': 'Rejeitar H0' if p_valor < alpha else 'Não rejeitar H0'
    }
    
    if p_valor < 0.001:
        resultado['nivel'] = "ALTAMENTE SIGNIFICATIVO"
    elif p_valor < 0.01:
        resultado['nivel'] = "MUITO SIGNIFICATIVO"
    elif p_valor < alpha:
        resultado['nivel'] = "SIGNIFICATIVO"
    else:
        resultado['nivel'] = "NÃO SIGNIFICATIVO"
    
    return resultado
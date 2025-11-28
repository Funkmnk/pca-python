"""
Módulo ETL - Extração, Transformação e Carregamento de Dados
Autor: Igor Chagas
Data: 26/11/2025

Responsabilidades:
- Carregamento de datasets CSV
- Sanitização obrigatória de colunas
- Validação de tipos de dados
- Limpeza e preparação inicial
"""

import pandas as pd
import numpy as np


def carregar_dataset_bruto(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega o dataset CSV e aplica sanitização básica obrigatória.
    
    SANITIZAÇÃO APLICADA:
    - Remove espaços em branco dos nomes das colunas
    - Valida que o arquivo foi carregado corretamente
    
    Args:
        caminho_arquivo: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com colunas sanitizadas
        
    Raises:
        FileNotFoundError: Se o arquivo não existir
        pd.errors.EmptyDataError: Se o arquivo estiver vazio
    """
    try:
        # Carregando o CSV
        df = pd.read_csv(caminho_arquivo)
        
        # Remove espaços em colunas
        df.columns = df.columns.str.strip()
        
        print(f"Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
        print(f"Colunas preparadas (espaços removidos)")
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Arquivo vazio: {caminho_arquivo}")


def validar_e_converter_tipos(df: pd.DataFrame, colunas_numericas: list = None) -> pd.DataFrame:
    """
    Converte colunas para tipos adequados, tratando erros de conversão.
    
    HIGIENE DE DADOS: Não confia na inferência automática do Pandas.
    
    Args:
        df: DataFrame original
        colunas_numericas: Lista de colunas que devem ser numéricas.
                          Se None, detecta automaticamente.
    
    Returns:
        DataFrame com tipos validados e convertidos
    """
    df_convertido = df.copy()
    
    if colunas_numericas is None:
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n" + "="*70)
    print("VALIDAÇÃO E CONVERSÃO DE TIPOS")
    print("="*70)
    
    for coluna in colunas_numericas:
        if coluna in df_convertido.columns:
            # Conversão com tratamento de erros
            df_convertido[coluna] = pd.to_numeric(df_convertido[coluna], errors='coerce')
            
            # valores não convertidos
            nulos_apos = df_convertido[coluna].isnull().sum()
            nulos_antes = df[coluna].isnull().sum()
            
            if nulos_apos > nulos_antes:
                print(f"{coluna}: {nulos_apos - nulos_antes} valores inválidos convertidos para NaN")
            else:
                print(f"{coluna}: Conversão numérica concluida!")
    
    return df_convertido


def carregar_dataset_processado(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega um dataset já processado (ex: com clusters, com discretização).
    
    Aplica apenas sanitização básica, sem validações pesadas.
    
    Args:
        caminho_arquivo: Caminho para o CSV processado
        
    Returns:
        DataFrame sanitizado
    """
    df = pd.read_csv(caminho_arquivo)
    df.columns = df.columns.str.strip()
    
    print(f"Dataset processado carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
    
    return df


def obter_features_numericas(df: pd.DataFrame, excluir_colunas: list = None) -> list:
    """
    Extrai lista de features numéricas, excluindo colunas especificadas.
    
    Args:
        df: DataFrame
        excluir_colunas: Lista de colunas para excluir (ex: ['ID', 'Target'])
        
    Returns:
        Lista com nomes das colunas numéricas
    """
    if excluir_colunas is None:
        excluir_colunas = []
    
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in colunas_numericas if col not in excluir_colunas]
    
    print(f"\nFeatures numéricas selecionadas: {len(features)}")
    print(f"  Excluídas: {excluir_colunas}")
    
    return features


def exibir_resumo_dataset(df: pd.DataFrame, titulo: str = "RESUMO DO DATASET"):
    """
    Exibe informações gerais sobre o dataset de forma narrativa.
    
    Args:
        df: DataFrame para análise
        titulo: Título do resumo
    """
    print("\n" + "="*70)
    print(f"{' ' * ((70 - len(titulo)) // 2)}{titulo}")
    print("="*70)
    
    print(f"\nDimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    
    # Análise de tipos
    print("\nTipos de dados:")
    tipos = df.dtypes.value_counts()
    for tipo, qtd in tipos.items():
        print(f"   - {tipo}: {qtd} colunas")
    
    # Análise de valores ausentes
    total_nulos = df.isnull().sum().sum()
    if total_nulos > 0:
        print(f"\nValores ausentes: {total_nulos:,} ({total_nulos / df.size * 100:.2f}% do total)")
        colunas_com_nulos = df.isnull().sum()[df.isnull().sum() > 0]
        print("\n   Colunas afetadas:")
        for col, qtd in colunas_com_nulos.items():
            print(f"   - {col}: {qtd} ({qtd / len(df) * 100:.1f}%)")
    else:
        print("\nSem valores ausentes no dataset!")
    
    print("="*70)
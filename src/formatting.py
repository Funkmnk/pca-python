"""
Módulo de Formatação - Utilitários para prints e organização visual
Autor: Igor Chagas
Data: 26/11/2024

Responsabilidades:
- Cabeçalhos e divisores padronizados
- Formatação de saída no terminal
"""


def montar_cabecalho(texto: str, largura: int = 70) -> None:
    """
    Imprime um cabeçalho centralizado com bordas.
    
    Args:
        texto: Texto do cabeçalho
        largura: Largura total do cabeçalho (padrão: 70)
    """
    print("\n" + "=" * largura)
    print(" " * ((largura - len(texto)) // 2) + texto)
    print("=" * largura + "\n")


def montar_divisor(texto: str, largura: int = 70) -> None:
    """
    Imprime um divisor de seção com texto centralizado.
    
    Args:
        texto: Texto do divisor
        largura: Largura total do divisor (padrão: 70)
    """
    print("\n" + "-" * largura)
    print(" " * ((largura - len(texto)) // 2) + texto)
    print("-" * largura + "\n")


def imprimir_progresso(etapa: str, total_etapas: int = None, etapa_atual: int = None) -> None:
    """
    Imprime progresso de forma narrativa.
    
    Args:
        etapa: Descrição da etapa atual
        total_etapas: Número total de etapas (opcional)
        etapa_atual: Número da etapa atual (opcional)
    """
    if total_etapas and etapa_atual:
        print(f"\n[{etapa_atual}/{total_etapas}] {etapa}...")
    else:
        print(f"\n-> {etapa}...")
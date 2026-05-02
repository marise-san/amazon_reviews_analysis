import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df) -> None:
    """
    Executa a Analise Exploratoria de Dados (EDA) sobre o DataFrame PySpark.
    Converte para Pandas para visualizacao dos graficos.
    """
    print("\n--- Iniciando Analise Exploratoria (EDA) ---")

    # Converte para Pandas para plotagem
    pandas_df = df.select("Review Text", "sentiment").limit(5000).toPandas()
    total = len(pandas_df)
    print(f"Total de registros analisados: {total}")

    sentiment_col = "sentiment"

    if sentiment_col not in pandas_df.columns:
        print("Atencao: Coluna 'sentiment' nao encontrada no Dataset.")
        print("\n--- Fim da Analise Exploratoria ---")
        return

    # Distribuicao de sentimentos
    sentiment_dist = pandas_df[sentiment_col].value_counts().reset_index()
    sentiment_dist.columns = [sentiment_col, "count"]

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=sentiment_dist,
        x=sentiment_col,
        y="count",
        hue=sentiment_col,
        palette="viridis",
        legend=False
    )
    plt.title("Distribuicao de Sentimentos nas Avaliacoes")
    plt.xlabel("Sentimento")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.show()

    print("\n--- Fim da Analise Exploratoria ---")

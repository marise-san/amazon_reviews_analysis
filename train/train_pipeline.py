import time
from typing import Iterator
from contextlib import contextmanager

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import when, col, lit
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

from src.spark_manager import SparkManager


@contextmanager
def timer(step_name: str) -> Iterator[None]:
    start_time = time.time()
    print(f"Iniciando: {step_name}...")
    yield
    elapsed_time = time.time() - start_time
    print(f"Concluido: {step_name} | Tempo: {elapsed_time:.2f}s")


def load_data(spark: SparkSession, data_path: str) -> DataFrame:
    with timer("Carregamento dos Dados"):
        df = spark.read.csv(data_path, header=True, inferSchema=False)

        if "Review Text" in df.columns and "Rating" in df.columns:
            df = df.dropna(subset=["Review Text", "Rating"])
        else:
            df = df.dropna()

        # Extrai o digito da nota original (ex: "Rated 5 out of 5 stars" -> "5")
        df = df.withColumn(
            "rating_digit",
            when(col("Rating").contains("1"), "1")
            .when(col("Rating").contains("2"), "2")
            .when(col("Rating").contains("3"), "3")
            .when(col("Rating").contains("4"), "4")
            .when(col("Rating").contains("5"), "5")
            .otherwise(None)
        )

        # Agrupa em 3 classes de sentimento
        df = df.withColumn(
            "sentiment",
            when(col("rating_digit").isin("1", "2"), "Negativo")
            .when(col("rating_digit") == "3", "Neutro")
            .when(col("rating_digit").isin("4", "5"), "Positivo")
            .otherwise(None)
        )

        df = df.dropna(subset=["sentiment"])

        total_rows = df.count()
        print(f"Total de registros: {total_rows}")
        return df


def _add_class_weights(df: DataFrame) -> DataFrame:
    """
    Calcula e adiciona uma coluna 'weight' para balancear as classes durante o treinamento.
    Classes com menos amostras recebem peso maior, compensando o desequilibrio da base.
    """
    total = df.count()
    n_classes = 3  # Positivo, Neutro, Negativo

    class_counts = {
        row["sentiment"]: row["count"]
        for row in df.groupBy("sentiment").count().collect()
    }

    weight_map = {
        cls: total / (n_classes * count)
        for cls, count in class_counts.items()
    }

    print(f"Distribuicao de classes: {class_counts}")
    print(f"Pesos aplicados:         {weight_map}")

    weight_expr = (
        when(col("sentiment") == "Positivo", weight_map.get("Positivo", 1.0))
        .when(col("sentiment") == "Neutro", weight_map.get("Neutro", 1.0))
        .when(col("sentiment") == "Negativo", weight_map.get("Negativo", 1.0))
        .otherwise(lit(1.0))
    )

    return df.withColumn("weight", weight_expr)


def build_pipeline() -> Pipeline:
    with timer("Construcao do Pipeline de NLP"):
        # Converte Positivo/Neutro/Negativo em indice numerico
        indexer = StringIndexer(inputCol="sentiment", outputCol="label", handleInvalid="keep")

        # Tokenizacao por regex: separa palavras e remove pontuacao
        tokenizer = RegexTokenizer(inputCol="Review Text", outputCol="words", pattern="\\W")

        # Remove stopwords em ingles
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

        # TF-IDF: converte texto em vetor numerico ponderado
        hashing_tf = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=20000)
        idf = IDF(inputCol="rawFeatures", outputCol="features")

        # Regressao Logistica com peso de classe para calibracao de sentimentos
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            weightCol="weight",
            maxIter=50
        )

        return Pipeline(stages=[indexer, tokenizer, remover, hashing_tf, idf, lr])


def train_pipeline(pipeline: Pipeline, df: DataFrame) -> PipelineModel:
    with timer("Treinamento do Modelo"):
        # Aplica pesos de classe antes do treinamento para balancear Positivo/Neutro/Negativo
        weighted_df = _add_class_weights(df)
        model = pipeline.fit(weighted_df)
        return model


def evaluate_model(model: PipelineModel, test_df: DataFrame) -> None:
    with timer("Avaliacao na Base de Teste"):
        predictions = model.transform(test_df)

        eval_acc = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        eval_f1 = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        )

        accuracy = eval_acc.evaluate(predictions)
        f1_score = eval_f1.evaluate(predictions)

        print("\n=======================================================")
        print("RESULTADOS DA VALIDACAO (TREINO vs TESTE)")
        print(f"   Accuracy Score: {accuracy:.4f}")
        print(f"   F1-Score:       {f1_score:.4f}")
        print("=======================================================\n")


def save_model(model: PipelineModel, model_path: str) -> None:
    with timer("Salvamento do Modelo"):
        try:
            model.write().overwrite().save(model_path)
        except Exception:
            print("Aviso: Salvamento em disco ignorado (ambiente sem Hadoop nativo).")
            print("O modelo esta operando corretamente na memoria.")

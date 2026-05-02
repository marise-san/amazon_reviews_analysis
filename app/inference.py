import os
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from app.schemas import ReviewInput, PredictionOutput


class SentimentModelService:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self._model = None

    def load_model(self) -> None:
        if self._model is not None:
            return

        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo nao encontrado: {self.model_path}")
        self._model = PipelineModel.load(self.model_path)

    def predict(self, spark: SparkSession, user_input: ReviewInput) -> PredictionOutput:
        if self._model is None:
            raise RuntimeError("Modelo nao carregado. Execute run_local_api.py primeiro.")

        safe_text = user_input.review_text.replace("'", "''")

        # Cria o DataFrame com as colunas esperadas pelo pipeline.
        # weight = 1.0 pois nao e utilizado durante a predicao, apenas no treinamento.
        df = spark.sql(
            f"SELECT '{safe_text}' AS `Review Text`, 'Neutro' AS sentiment, 1.0 AS weight"
        )

        predictions = self._model.transform(df)
        result = predictions.select("prediction", "probability").first()

        if result is None:
            raise RuntimeError("Nao foi possivel gerar a previsao.")

        prediction_val = float(result["prediction"])
        prob_vector = result["probability"]
        probability_val = float(prob_vector[int(prediction_val)])

        # Recupera o label original do StringIndexer (Positivo, Neutro ou Negativo)
        indexer_model = self._model.stages[0]
        sentimento = indexer_model.labels[int(prediction_val)]

        confianca_pct = f"{probability_val * 100:.2f}%"
        msg = f"O modelo classificou esta avaliacao como {sentimento} com {confianca_pct} de confianca."

        return PredictionOutput(
            prediction_id=int(prediction_val),
            sentiment=sentimento,
            confidence_percentage=confianca_pct,
            human_readable_message=msg
        )

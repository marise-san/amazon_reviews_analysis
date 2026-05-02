from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException

from app.schemas import ReviewInput, PredictionOutput
from app.inference import SentimentModelService
from src.spark_manager import SparkManager

MODEL_PATH = "models/amazon_sentiment_model"
sentiment_service = SentimentModelService(model_path=MODEL_PATH)
spark_manager = SparkManager()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("Iniciando contexto da API...")
    spark_manager.get_spark_session("FastAPI_Inferencia")

    try:
        sentiment_service.load_model()
        print("Modelo carregado na memoria com sucesso.")
    except Exception as e:
        print(f"Aviso: {e}")

    yield

    spark_manager.stop_session()
    print("API desligada.")

app = FastAPI(
    title="Amazon Reviews Sentiment API",
    description="API de análise de sentimentos com PySpark ML. Classifica avaliações como Positivo, Neutro ou Negativo.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs"
)

@app.post("/predict", response_model=PredictionOutput)
async def predict_sentiment(review: ReviewInput) -> PredictionOutput:
    spark = spark_manager.get_spark_session()

    try:
        return sentiment_service.predict(spark=spark, user_input=review)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn
import glob
import logging
import signal

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from app.main import app, sentiment_service
from src.spark_manager import SparkManager
from src.downloader import KaggleDownloader
from train.train_pipeline import load_data, build_pipeline, train_pipeline, evaluate_model


def start_api():
    print("\n=======================================================")
    print(" Amazon Reviews - API de Analise de Sentimentos")
    print("=======================================================\n")

    # 1. Localizar dados
    print("[1/3] Localizando base de dados do Kaggle...")
    path = KaggleDownloader.download_dataset("dongrelaxman/amazon-reviews-dataset")
    csv_files = glob.glob(f"{path}/*.csv")
    main_csv = csv_files[0] if csv_files else path

    # 2. Inicializar Spark e treinar o modelo
    print("\n[2/3] Inicializando Spark e treinando o modelo...")
    spark = SparkManager().get_spark_session("AmazonSentimentAPI")
    spark.sparkContext.setLogLevel("ERROR")

    df = load_data(spark, main_csv)

    print("\n[2.1] Dividindo base: 80% treino / 20% teste...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    pipeline = build_pipeline()
    model = train_pipeline(pipeline, train_df)

    print("\n[2.2] Avaliando modelo na base de teste...")
    evaluate_model(model, test_df)

    # 3. Injetar modelo na API e subir servidor
    print("[3/3] Carregando modelo na API...")
    sentiment_service._model = model

    print("\n=======================================================")
    print(" API PRONTA! Acesse: http://127.0.0.1:8000/docs")
    print("=======================================================\n")

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    start_api()

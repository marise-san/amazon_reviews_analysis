import os
import sys
import threading
from typing import Optional
from pyspark.sql import SparkSession


class SparkManager:
    """Singleton que gerencia o ciclo de vida da SparkSession."""

    _instance: Optional["SparkManager"] = None
    _lock: threading.Lock = threading.Lock()
    _spark: Optional[SparkSession] = None

    def __new__(cls) -> "SparkManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SparkManager, cls).__new__(cls)
        return cls._instance

    def get_spark_session(self, app_name: str = "AmazonReviewsApp") -> SparkSession:
        if self._spark is None:
            os.environ["PYSPARK_PYTHON"] = sys.executable
            os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

            print(f"Inicializando SparkSession para: {app_name}...")

            # Suprime os warnings de configuracao do Hadoop durante a inicializacao
            null_fd = os.open(os.devnull, os.O_RDWR)
            save_fd = os.dup(2)
            os.dup2(null_fd, 2)

            try:
                self._spark = (
                    SparkSession.builder
                    .appName(app_name)
                    .master("local[*]")
                    .config("spark.driver.memory", "4g")
                    .config("spark.executor.memory", "4g")
                    .config("spark.sql.shuffle.partitions", "10")
                    .getOrCreate()
                )
            finally:
                os.dup2(save_fd, 2)
                os.close(null_fd)

            print("SparkSession inicializada com sucesso.")
        return self._spark

    def stop_session(self) -> None:
        if self._spark is not None:
            try:
                self._spark.stop()
            except Exception:
                pass
            self._spark = None
            print("SparkSession encerrada.")

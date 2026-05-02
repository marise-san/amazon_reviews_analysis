import kagglehub

class KaggleDownloader:
    @staticmethod
    def download_dataset(dataset_identifier: str) -> str:
        """
        Faz o download da base de dados via KaggleHub.
        O KaggleHub gerencia o cache local, evitando downloads duplicados.
        """
        print(f"Acionando API do Kaggle para a base: {dataset_identifier}...")
        path = kagglehub.dataset_download(dataset_identifier)
        print(f"Base de dados disponivel localmente em: {path}")
        return path

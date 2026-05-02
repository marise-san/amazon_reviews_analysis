import requests
import sys


def test_prediction():
    url = "http://localhost:8000/predict"
    payload = {
        "review_text": "This product is absolutely amazing, I loved it!"
    }

    print("Enviando requisicao de teste para a API...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        print("\n--- Resultado da Previsao ---")
        print(f"Sentimento Analisado: {data.get('sentiment')}")
        print(f"Grau de Confianca:    {data.get('confidence_percentage')}")
        print(f"Mensagem:             {data.get('human_readable_message')}")
        print("-----------------------------\n")
        print("Teste concluido com sucesso!")

    except requests.exceptions.ConnectionError:
        print("ERRO: A API nao esta rodando. Execute 'python run_local_api.py' primeiro.")
        sys.exit(1)
    except Exception as e:
        print(f"Erro inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_prediction()

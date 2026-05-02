# Análise de Sentimentos — Amazon Reviews

Projeto de Big Data desenvolvido como trabalho final do MBA, implementando um pipeline completo de Machine Learning com **PySpark** e exposição via **FastAPI**.

O sistema processa avaliações de produtos da Amazon, treina um modelo de NLP distribuído e disponibiliza uma API REST para **análise de sentimentos** em tempo real, classificando textos como **Positivo**, **Neutro** ou **Negativo**.

## Estrutura do Projeto

```
amazon_reviews_analysis/
├── app/                  # API FastAPI (endpoint de predição)
│   ├── inference.py      # Lógica de inferência do modelo
│   ├── main.py           # Configuração da aplicação FastAPI
│   └── schemas.py        # Modelos de entrada e saída (Pydantic)
├── src/                  # Módulos auxiliares
│   ├── spark_manager.py  # Gerenciamento do ciclo de vida do Spark
│   ├── downloader.py     # Download automático da base via KaggleHub
│   └── eda.py            # Análise exploratória dos dados
├── train/
│   └── train_pipeline.py # Pipeline de NLP e treinamento (TF-IDF + Regressão Logística)
├── test/
│   └── test_api.py       # Script de teste automatizado da API
├── apresentacao_mba.ipynb  # Notebook de apresentação do projeto
├── run_local_api.py        # Script principal: treina e sobe a API
└── requirements.txt
```

## Pipeline de Machine Learning

O modelo é construído com o seguinte pipeline de NLP:

1. **StringIndexer** — converte a coluna `sentiment` (Positivo/Neutro/Negativo) em índice numérico (label)
2. **RegexTokenizer** — tokeniza o texto removendo pontuações
3. **StopWordsRemover** — remove palavras sem valor semântico (the, is, at...)
4. **HashingTF + IDF** — converte texto em vetor numérico ponderado por frequência
5. **LogisticRegression** — classificador multiclasse com balanceamento de classes

As notas originais da Amazon (1 a 5 estrelas) são agrupadas em **3 classes de sentimento** antes do treinamento:
- Notas 1–2 → **Negativo**
- Nota 3 → **Neutro**
- Notas 4–5 → **Positivo**

A base de dados é dividida em **80% treino / 20% teste** com `randomSplit` e as métricas de **Accuracy** e **F1-Score** são calculadas e exibidas no terminal a cada execução.

## Pré-requisitos

- Python 3.10+
- Java 11+ (necessário para o PySpark)
- Token da API do Kaggle configurado como variável de ambiente `KAGGLE_API_TOKEN`

### Instalação

```bash
pip install -r requirements.txt
```

## Como Executar

Execute o script principal na raiz do projeto:

```bash
python run_local_api.py
```

O script irá:
1. Baixar automaticamente a base de dados da Amazon via KaggleHub
2. Inicializar o PySpark e treinar o modelo
3. Exibir as métricas de validação (Accuracy e F1-Score)
4. Subir a API REST na porta `8000`

Quando o servidor estiver pronto, acesse a interface de testes:

```
http://127.0.0.1:8000/docs
```

## Testando a API

### Via Swagger (recomendado)

Acesse `http://127.0.0.1:8000/docs`, clique em `POST /predict` → `Try it out`, insira o JSON e clique em `Execute`.

**Exemplos de entrada e saída esperada:**

---

**Avaliação positiva:**
```json
{ "review_text": "I absolutely loved this product, it works perfectly!" }
```
```json
{
  "prediction_id": 0,
  "sentiment": "Positivo",
  "confidence_percentage": "84.31%",
  "human_readable_message": "O modelo classificou esta avaliacao como Positivo com 84.31% de confianca."
}
```

---

**Avaliação negativa:**
```json
{ "review_text": "Terrible product, completely broken on arrival. Waste of money." }
```
```json
{
  "prediction_id": 1,
  "sentiment": "Negativo",
  "confidence_percentage": "76.22%",
  "human_readable_message": "O modelo classificou esta avaliacao como Negativo com 76.22% de confianca."
}
```

---

**Avaliação neutra/mista:**
```json
{ "review_text": "Nice product, but the delivery was really slow." }
```
```json
{
  "prediction_id": 2,
  "sentiment": "Neutro",
  "confidence_percentage": "51.40%",
  "human_readable_message": "O modelo classificou esta avaliacao como Neutro com 51.40% de confianca."
}
```
> Avaliações mistas tendem a ter confiança menor, o que é esperado em problemas de NLP com ambiguidade semântica.

---

### Via terminal (teste automatizado)

Com a API rodando, abra um segundo terminal e execute:

```bash
python test/test_api.py
```

## Execução via Docker

Caso prefira rodar o projeto em container, sem precisar configurar o ambiente Python e Java localmente:

**1. Construir a imagem:**
```bash
docker build -t amazon-sentiment .
```

**2. Executar o container** passando o token do Kaggle como variável de ambiente:
```bash
docker run -e KAGGLE_API_TOKEN=seu_token_aqui -p 8000:8000 amazon-sentiment
```

O container irá automaticamente baixar a base de dados, treinar o modelo e subir a API na porta `8000`.

**3. Testar:**

Acesse `http://localhost:8000/docs` no seu navegador.

> **Nota:** A variável `KAGGLE_API_TOKEN` é obrigatória para o download automático da base. Nunca inclua o token diretamente no código ou no Dockerfile.

---

## Tecnologias Utilizadas

| Tecnologia | Versão | Função |
|---|---|---|
| PySpark | 3.x | Processamento distribuído e ML |
| FastAPI | 0.x | API REST assíncrona |
| KaggleHub | latest | Download automático da base |
| Uvicorn | 0.x | Servidor ASGI |
| Pydantic | 2.x | Validação de dados |
| Docker | - | Containerização para deploy |

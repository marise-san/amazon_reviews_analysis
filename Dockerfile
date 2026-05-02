FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Java é obrigatório para o PySpark
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# O script run_local_api.py treina o modelo na memória e sobe o servidor
# A variável KAGGLE_API_TOKEN deve ser passada via: docker run -e KAGGLE_API_TOKEN=...
CMD ["python", "run_local_api.py"]

FROM ghcr.io/mlflow/mlflow:v2.16.1

# Dodaj driver do Postgresa
RUN pip install --no-cache-dir psycopg2-binary

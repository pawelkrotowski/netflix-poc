FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Podstawowe narzędzia + Python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Aktualizacja pip i instalacja paczek (PyTorch cu121)
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      jupyter jupyterlab matplotlib pandas numpy scipy scikit-learn \
      sqlalchemy psycopg2-binary mlflow optuna fastapi uvicorn[standard] plotly \
      implicit \
      --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
      --extra-index-url https://download.pytorch.org/whl/cu121

ENV PYTHONUNBUFFERED=1

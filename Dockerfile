FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        torch==2.14.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install \
        -r requirements-runtime.txt

COPY src ./src
COPY scripts/predict_segmentation.py ./scripts/predict_segmentation.py
COPY configs/segformer_b0_binary_config.json \
     ./configs/segformer_b0_binary_config.json

RUN mkdir -p \
    /app/models \
    /input \
    /output

ENTRYPOINT ["python", "-m", "scripts.predict_segmentation"]

#FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    TRANSFORMERS_CACHE=/workspace/cache/transformers \
    HF_HOME=/workspace/cache/huggingface \
    DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app /workspace/cache/transformers /workspace/cache/huggingface \
    && chmod -R 777 /workspace/cache # Ensure cache dirs are writable

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# This expects the config file path to be passed as an argument at runtime
CMD ["python", "src/run_experiment.py"]
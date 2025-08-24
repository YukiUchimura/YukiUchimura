FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/models_hf \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# g++等のビルド必須ツール + ランタイム
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential gcc g++ make cmake \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1080Ti 向けに CUDA 11.8 の PyTorch を固定導入（torchvisionと同時）
RUN python -m pip install --upgrade "pip<24.1" \
 && pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Python 依存
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# アプリ本体
COPY . /workspace

EXPOSE 7860
CMD ["python","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","7860"]

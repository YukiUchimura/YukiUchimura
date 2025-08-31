FROM python:3.10-slim

ARG CUDA_VER=cu118
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    DIFFUSERS_CACHE=/workspace/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 必要ツール & ビルドに必要なヘッダ類（insightfaceでg++必須）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential g++ cmake \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 先にTorch(CUDA11.8)を固定導入
# 1080 Ti でも 11.x 系CUDAで動作します（ドライバは新しめ必須）
RUN python -m pip install --upgrade "pip<25.3" setuptools wheel && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
      torch==2.1.0+cu118 torchvision==0.16.0+cu118

# Python依存
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# プロジェクト一式
COPY . /workspace

# モデルの事前取得（トークンがあればログイン→ダウンロード）
RUN bash ./setup.sh

EXPOSE 8000
CMD ["bash","-lc","uvicorn app.api:app --host 0.0.0.0 --port 8000"]

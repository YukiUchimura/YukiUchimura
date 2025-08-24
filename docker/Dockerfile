FROM python:3.10-slim

# 基本ツール
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 依存ファイルを先にコピー（キャッシュ効かせる）
COPY requirements.txt /workspace/requirements.txt

# --- 重要 ---
# GTX 1080 Ti は CUDA 11.x 系を利用できます。ここでは cu118 を採用。
# PyTorch + CUDA 11.8 を先に入れた後、他依存を入れます。
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# アプリ本体コピー
COPY . /workspace

# モデル・入出力用ディレクトリ
RUN mkdir -p /workspace/models_hf /workspace/uploads /workspace/outputs

# コンテナ起動時に FastAPI を立ち上げ
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

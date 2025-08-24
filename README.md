# Ultimate Face Synthesis (SDXL + InstantID + IP-Adapter + OpenPose)

## 必要環境
- **OS**: Windows 10/11, macOS, or Linux
- **Docker & Docker Compose**: Docker Desktop or Docker Engine.
- **GPU**: NVIDIA GPU with at least 11GB VRAM (e.g., GTX 1080 Ti) is highly recommended.
- **NVIDIA Drivers**: Ensure you have NVIDIA drivers that support CUDA 11.x.
- **PyTorch Version**: The setup uses `torch==2.1.0+cu118`, which requires a compatible NVIDIA driver.
- **NVIDIA Container Toolkit**: Required for Linux to use GPUs with Docker. For Docker Desktop (Windows/Mac), GPU support should be enabled in the settings.

## セットアップ（共通）

1.  **リポジトリをクローン**
    ```bash
    git clone <THIS_REPO_URL>
    cd <cloned_dir>
    ```

2.  **Hugging Face トークンの設定 (任意)**
    一部のモデルは、ダウンロードに Hugging Face の認証が必要です。必要に応じて、プロジェクトルートに `.env` ファイルを作成し、アクセストークンを記述してください。
    ```
    # .env ファイルの中身
    HUGGING_FACE_HUB_TOKEN=hf_YOUR_ACCESS_TOKEN_HERE
    ```

3.  **コンテナのビルドと起動**
    以下のコマンドでコンテナをビルドし、起動します。初回起動時は、数GBのモデルファイルが `./models_hf` にダウンロードされるため、時間がかかります。
    ```bash
    docker compose build
    docker compose up
    ```
    コンテナを停止するには `docker compose down` を実行します。

### Docker Desktop でGPUが認識されない場合
Docker Desktop の設定でGPU共有が有効になっていても `compose` コマand でGPUが使えない場合があります。その場合は、`docker run` で直接GPUを指定して起動してみてください。

**Windows (PowerShell):**
```powershell
docker compose build
docker run --rm -it --gpus all -p 7860:7860 `
  -v ${pwd}/models_hf:/workspace/models_hf `
  -v ${pwd}/uploads:/workspace/uploads `
  -v ${pwd}/outputs:/workspace/outputs `
  -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} `
  --name face-synthesis-container `
  face_synthesis_api
```

**macOS / Linux (Bash):**
```bash
docker compose build
docker run --rm -it --gpus all -p 7860:7860 \
  -v $(pwd)/models_hf:/workspace/models_hf \
  -v $(pwd)/uploads:/workspace/uploads \
  -v $(pwd)/outputs:/workspace/outputs \
  -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} \
  --name face-synthesis-container \
  face_synthesis_api
```


## APIの使い方

### FastAPI のドキュメント (Swagger UI)
コンテナ起動後、ブラウザで `http://localhost:7860/docs` を開くと、APIの仕様を確認し、直接ファイルをアップロードして試すことができます。

- `face_images`: 1〜5枚の顔写真（PNG/JPEG）
- `style_image`: 雰囲気（背景・色合い）の参照画像
- `pose_image`: (任意) ポーズの参照画像
- 各種パラメータ (`prompt`, `face_scale`, `style_scale` 等) を調整

### curl を使った例
コマンドラインからAPIを叩く場合の例です。

```bash
# me1.jpg, me2.jpg, cafe.jpg, pose.jpg がカレントディレクトリにあると仮定
curl -X POST "http://localhost:7860/generate/" \
  -F "face_images=@me1.jpg" \
  -F "face_images=@me2.jpg" \
  -F "style_image=@cafe.jpg" \
  -F "pose_image=@pose.jpg" \
  -F "prompt=a candid photo in a cozy cafe" \
  -F "negative_prompt=low quality, artifacts" \
  -F "face_scale=1.0" \
  -F "style_scale=0.6" \
  -F "steps=28" \
  --output generated_image.png
```

## メモリ調整のヒント

- **VRAM不足**: 1024x1024 の画像生成はVRAMを多く消費します。`PYTORCH_CUDA_ALLOC_CONF` を `docker-compose.yml` に設定済みですが、それでもエラーが出る場合は、`app/image_generator/pipeline.py` 内の解像度を 768 や 512 に変更してみてください。
- **ステップ数**: `steps` パラメータは20〜30程度から試すのがおすすめです。
- **同時リクエスト**: 複数のリクエストを同時に処理するとVRAMが不足する可能性があるため、1つずつ実行してください。

## フォルダ構成
- `models_hf/`: Hugging Face からダウンロードされたモデルのキャッシュ
- `uploads/`: API経由でアップロードされた一時ファイル
- `outputs/`: (現状未使用) 将来的に生成画像を保存するためのフォルダ

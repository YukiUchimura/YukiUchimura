# Ultimate Face Synthesis (SDXL + InstantID + IP-Adapter + OpenPose)

## 必要環境
- Windows 10/11 + Docker Desktop（GPU共有をON）
- NVIDIA ドライバ（CUDA 11.x 対応）
- NVIDIA Container Toolkit（Docker から GPU を使う場合）
- GPU: GTX 1080 Ti（VRAM 11GB推奨）

## 事前確認
- `nvidia-smi` に GPU が表示されること
- Docker Desktop の Settings → Resources → GPU を有効化

## セットアップ（F:\face_synthesis）
1. このフォルダ構成のまま保存
2. 必要なら `.env` に `HUGGING_FACE_HUB_TOKEN` を設定
3. 初回起動
   ```powershell
   cd F:\face_synthesis
   docker compose build
   docker compose up
   ```

初回実行時はモデルをダウンロードするため時間が掛かります。

## 使い方

ブラウザで http://localhost:7860/docs を開く

/generate/ に以下をアップロード

- `face_images`: 1〜5枚の顔写真（PNG/JPEG）
- `style_image`: 雰囲気（背景・色合い）参照画像
- `pose_image`: 任意。ある場合はポーズを反映

`face_scale`（本人らしさ）と `style_scale`（雰囲気強度）、`steps`、`seed` を必要に応じて調整

実行するとPNG画像が返ってきます

## メモリ調整のヒント

- 1024x1024 が重いときは 768 や 512 に変更（pipeline内を調整）
- ステップ数は 20〜30 程度から
- 同時並行リクエストは避ける

## フォルダ

- `models_hf/`: モデルキャッシュ（再ダウンロード防止）
- `uploads/`: 一時アップロード
- `outputs/`: 任意で保存先に利用（現状はAPI返却のみ）

---

# 3) 実行手順（Windows / 1080Ti）

1. **NVIDIA ドライバ & Docker Desktop** を導入。Docker Desktop の設定で **GPU 共有** を ON。
2. **NVIDIA Container Toolkit** を入れておくとより確実（Docker から GPU を使えるようにする）。
3. 上記ファイルを**そのまま** `F:\face_synthesis` に保存。
4. PowerShell で:
   ```powershell
   cd F:\face_synthesis
   docker compose build
   docker compose up
   ```

初回は モデルが `F:\face_synthesis\models_hf` にキャッシュされます。

ブラウザで `http://localhost:7860/docs` を開き、`/generate/` に顔写真 1～5枚、雰囲気画像、（任意で）ポーズ画像を入れて実行。

返却された PNG が合成結果です。

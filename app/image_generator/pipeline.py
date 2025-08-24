"""
顔画像・雰囲気画像・ポーズ画像を組み合わせて一枚の高品質な画像を生成する
パイプライン実装です。

- InstantID: 顔特徴を抽出し、本人らしさを維持
- IP-Adapter: 雰囲気画像の色合いや世界観を反映
- OpenPose ControlNet: ポーズ情報を抽出・適用
- SDXL: 高品質な最終画像を生成

初回のみモデルをダウンロードして /workspace/models_hf にキャッシュします。
"""

import os
import traceback
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    EulerDiscreteScheduler,
)
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from controlnet_aux.open_pose import OpenposeDetector
import cv2

# 環境変数からモデルキャッシュディレクトリを取得
MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "models_hf")

# グローバルモデル
face_analysis_app: Optional[FaceAnalysis] = None
openpose_detector: Optional[OpenposeDetector] = None
sdxl_pipe: Optional[StableDiffusionXLControlNetPipeline] = None

# モデルID
SDXL_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
INSTANTID_CONTROLNET_ID = "InstantX/InstantID"
OPENPOSE_CONTROLNET_ID = "thibaud/controlnet-openpose-sdxl-1.0"
IP_ADAPTER_ID = "h94/IP-Adapter"
IP_ADAPTER_WEIGHTS_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sdxl.bin"


def load_models() -> None:
    """必要なすべてのモデルをロードしてキャッシュする。"""
    global face_analysis_app, openpose_detector, sdxl_pipe

    if sdxl_pipe is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        # FaceAnalysis (InstantID 用)
        print("[load_models] Loading FaceAnalysis (InstantID) ...")
        face_analysis_app = FaceAnalysis(
            name="buffalo_l",
            root=MODEL_CACHE_DIR,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        face_analysis_app.prepare(ctx_id=0, det_size=(640, 640))
        print("[load_models] FaceAnalysis loaded.")

        # OpenPose Detector
        print("[load_models] Loading OpenposeDetector ...")
        try:
            openpose_detector = OpenposeDetector.from_pretrained(
                "lllyasviel/ControlNet", cache_dir=MODEL_CACHE_DIR
            )
            print("[load_models] OpenposeDetector loaded.")
        except Exception as e:
            print(f"[load_models] Failed to load OpenPose annotator: {e}")
            print("This may be due to a network issue, a firewall/proxy, or a missing Hugging Face token.")
            print("Please check your network connection and credentials.")
            raise


        # ControlNet (InstantID, OpenPose)
        print("[load_models] Loading ControlNet models ...")
        controlnet_instantid = ControlNetModel.from_pretrained(
            INSTANTID_CONTROLNET_ID,
            subfolder="controlnet",
            torch_dtype=torch_dtype,
            cache_dir=MODEL_CACHE_DIR,
        )
        controlnet_openpose = ControlNetModel.from_pretrained(
            OPENPOSE_CONTROLNET_ID,
            torch_dtype=torch_dtype,
            cache_dir=MODEL_CACHE_DIR,
        )
        controlnets = [controlnet_instantid, controlnet_openpose]
        print("[load_models] ControlNet models loaded.")

        # SDXL + ControlNet
        print(f"[load_models] Loading SDXL pipeline: {SDXL_BASE_MODEL_ID} ...")
        sdxl_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE_MODEL_ID,
            controlnet=controlnets,
            torch_dtype=torch_dtype,
            cache_dir=MODEL_CACHE_DIR,
        )
        sdxl_pipe.scheduler = EulerDiscreteScheduler.from_config(
            sdxl_pipe.scheduler.config
        )

        # IP-Adapter
        print("[load_models] Loading IP-Adapter ...")
        sdxl_pipe.load_ip_adapter(
            IP_ADAPTER_ID,
            subfolder=IP_ADAPTER_WEIGHTS_SUBFOLDER,
            weight_name=IP_ADAPTER_WEIGHT_NAME,
            cache_dir=MODEL_CACHE_DIR,
        )
        sdxl_pipe.to(device)
        print("[load_models] All models loaded successfully.")
    except Exception:
        print("[load_models] Error while loading models.")
        traceback.print_exc()
        raise


def _compute_face_embeddings(face_image_paths: List[str]) -> torch.Tensor:
    """複数枚の顔写真から InstantID 用の平均埋め込みを作る。"""
    global face_analysis_app
    embeddings: List[np.ndarray] = []

    for path in face_image_paths:
        with open(path, "rb") as f:
            img_bytes = np.frombuffer(f.read(), np.uint8)
        cv2_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if cv2_img is None:
            continue

        faces = face_analysis_app.get(cv2_img)
        if not faces:
            continue

        faces = sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True,
        )
        embeddings.append(faces[0].embedding)

    if not embeddings:
        raise ValueError("No faces detected in the provided face images.")

    avg_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
    return torch.tensor(avg_embedding, dtype=torch.float32).unsqueeze(0)


def generate_unified_image(
    face_image_paths: List[str],
    style_image_path: str,
    pose_image_path: Optional[str],
    prompt: str,
    negative_prompt: str,
    face_scale: float,
    style_scale: float,
    num_inference_steps: int,
    seed: Optional[int],
) -> Image.Image:
    """顔+雰囲気+ポーズから一枚の画像を生成。"""
    global sdxl_pipe, openpose_detector

    if sdxl_pipe is None:
        raise RuntimeError("Models are not loaded. Call load_models() first.")

    # 1) 顔埋め込み
    print("[generate_unified_image] Computing face embeddings ...")
    face_embeddings = _compute_face_embeddings(face_image_paths).to(sdxl_pipe.device)

    # 2) 雰囲気画像 + IP-Adapter 強度
    print("[generate_unified_image] Loading style image and setting IP-Adapter scale ...")
    style_image = load_image(style_image_path).resize((1024, 1024))
    sdxl_pipe.set_ip_adapter_scale(style_scale)

    # 3) ControlNet 入力（InstantID 用の顔画像、OpenPose の骨格画像）
    control_images: List[Image.Image] = []
    control_scales: List[float] = []

    # InstantID 用：顔画像そのものを ControlNet 入力として使う
    face_control_image = load_image(face_image_paths[0])
    control_images.append(face_control_image)
    control_scales.append(face_scale)

    # OpenPose
    if pose_image_path:
        print("[generate_unified_image] Generating pose keypoints with OpenPose ...")
        pose_img = load_image(pose_image_path)
        pose_key = openpose_detector(
            pose_img, detect_resolution=512, image_resolution=1024
        )
        control_images.append(pose_key)
        control_scales.append(0.8)  # 必要に応じてパラメータ化可
    else:
        control_images.append(Image.new("RGB", (1024, 1024)))
        control_scales.append(0.0)

    # 4) 乱数シード
    generator = None
    if seed is not None:
        generator = torch.Generator(device=sdxl_pipe.device).manual_seed(seed)

    # 5) 生成実行
    print("[generate_unified_image] Running SDXL pipeline ...")
    try:
        out = sdxl_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_images,
            controlnet_conditioning_scale=control_scales,
            ip_adapter_image=style_image,
            prompt_embeds=face_embeddings,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.0,
            generator=generator,
        )
        img = out.images[0]
    except Exception:
        print("[generate_unified_image] Error during generation.")
        traceback.print_exc()
        raise

    print("[generate_unified_image] Done.")
    return img

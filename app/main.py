import io
import os
import uuid
import traceback
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from app.image_generator.pipeline import load_models, generate_unified_image

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Ultimate Face Synthesis Generator",
    description="Combine your face, a style photo, and an optional pose to generate a highly realistic composite (SDXL + InstantID + IP-Adapter + OpenPose).",
    version="1.0.0",
)

@app.on_event("startup")
async def on_startup() -> None:
    print("[main] Loading models on startup ...")
    try:
        load_models()
        print("[main] Models loaded.")
    except Exception as e:
        print("[main] Model load failed!")
        traceback.print_exc()
        raise RuntimeError(f"Could not load models: {e}")

def _save_temp(upload: UploadFile) -> str:
    try:
        fname = f"{uuid.uuid4()}{os.path.splitext(upload.filename)[1]}"
        path = os.path.join(UPLOAD_DIR, fname)
        with open(path, "wb") as f:
            f.write(upload.file.read())
        return path
    finally:
        upload.file.close()

@app.post("/generate/")
async def generate(
    face_images: List[UploadFile] = File(..., description="1 to 5 face images (PNG/JPEG)"),
    style_image: UploadFile = File(..., description="A style/mood reference image (PNG/JPEG)"),
    pose_image: Optional[UploadFile] = File(None, description="Optional pose reference (PNG/JPEG)"),
    prompt: str = Form("a photo of a person"),
    negative_prompt: str = Form("deformed, disfigured, low quality, blurry, ugly"),
    face_scale: float = Form(1.0, ge=0.0, le=2.0),
    style_scale: float = Form(0.6, ge=0.0, le=1.0),
    steps: int = Form(30, ge=10, le=100),
    seed: Optional[int] = Form(None),
) -> StreamingResponse:
    if not 1 <= len(face_images) <= 5:
        raise HTTPException(status_code=400, detail="Please upload between 1 and 5 face images.")

    tmp: List[str] = []
    try:
        face_paths = []
        for f in face_images:
            p = _save_temp(f)
            face_paths.append(p)
            tmp.append(p)
        s_path = _save_temp(style_image)
        tmp.append(s_path)
        p_path = None
        if pose_image is not None:
            p_path = _save_temp(pose_image)
            tmp.append(p_path)

        img: Image.Image = generate_unified_image(
            face_image_paths=face_paths,
            style_image_path=s_path,
            pose_image_path=p_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            face_scale=face_scale,
            style_scale=style_scale,
            num_inference_steps=steps,
            seed=seed,
        )

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")
    finally:
        for p in tmp:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

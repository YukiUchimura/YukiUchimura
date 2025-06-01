from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import StreamingResponse
import io
import zipfile
from typing import List, Optional, Dict
import os
import shutil
import uuid
import time # For metadata timestamp
import json # For metadata saving
from pydantic import BaseModel

# Assuming pipeline.py is in app/image_generator/
from .image_generator.pipeline import generate_image_with_instantid, load_models as load_pipeline_models

# --- Configuration & Directories ---
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

app = FastAPI(title="SDXL InstantID Generator API")

# --- Global State (Simplification for this example) ---
# In a production app, this might be a database or a more robust cache
uploaded_file_paths: Dict[str, str] = {}

# --- Application Startup ---
@app.on_event("startup")
async def load_models_on_startup():
    print("FastAPI application startup: Loading ML models...")
    try:
        load_pipeline_models() # This is a synchronous call from pipeline.py
        print("ML models loaded successfully.")
    except Exception as e:
        print(f"Error loading ML models on startup: {e}")
        # Depending on severity, you might want to prevent app startup or handle differently
        # For now, just log the error. The pipeline functions will raise errors if models aren't loaded.

# --- Helper Functions ---
async def save_upload_file(upload_file: UploadFile, destination: str) -> None:
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

# --- Request Models ---
class GenerateImageParams(BaseModel):
    face_image_ids: List[str]
    style_image_id: str
    pose_image_id: Optional[str] = None
    prompt: str = "A photo of a person, cinematic lighting" # Added prompt

    # Scales matching the pipeline function
    steps: int = 30 # Renamed from num_inference_steps for simplicity in API
    guidance_scale: float = 5.0
    instantid_scale: float = 0.80
    style_shuffle_scale: float = 0.70
    pose_control_scale: float = 0.75
    ip_adapter_scale: float = 0.80 # For InstantID identity strength
    seed: Optional[int] = None # Allow seed to be optional, pipeline might generate one

# --- Background Task for Generation ---
def generate_image_task(
    output_image_id: str,
    face_image_paths_resolved: List[str],
    style_image_path_resolved: str,
    pose_image_path_resolved: Optional[str],
    params: GenerateImageParams
):
    print(f"Background task started for output_image_id: {output_image_id}")
    start_time = time.time()

    # Prepare a default seed if None (pipeline might also do this)
    # For reproducibility, it's good to record the seed used.
    actual_seed = params.seed if params.seed is not None else int(time.time()) # Simple default seed

    try:
        generated_image = generate_image_with_instantid(
            face_image_paths=face_image_paths_resolved,
            style_image_path=style_image_path_resolved,
            prompt=params.prompt,
            pose_image_path=pose_image_path_resolved,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance_scale,
            instantid_scale=params.instantid_scale,
            style_shuffle_scale=params.style_shuffle_scale,
            pose_control_scale=params.pose_control_scale,
            ip_adapter_scale=params.ip_adapter_scale
            # Seed handling: The pipeline function itself doesn't explicitly take a seed yet.
            # This would require modification of `generate_image_with_instantid` to accept and use a generator.
            # For now, this seed isn't passed to the core pipeline.
        )

        image_save_path = os.path.join(IMAGE_DIR, f"{output_image_id}.png")
        generated_image.save(image_save_path)
        print(f"Generated image saved to {image_save_path}")

        end_time = time.time()
        generation_duration = end_time - start_time

        metadata = {
            "output_image_id": output_image_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generation_duration_seconds": round(generation_duration, 2),
            "input_params": params.dict(),
            "resolved_paths": {
                "face_images": face_image_paths_resolved,
                "style_image": style_image_path_resolved,
                "pose_image": pose_image_path_resolved if pose_image_path_resolved else "Not provided"
            },
            "status": "completed",
            "seed_used": actual_seed # Placeholder, as pipeline doesn't use it yet
        }

    except Exception as e:
        print(f"Error during image generation task for {output_image_id}: {e}")
        import traceback
        traceback.print_exc()
        metadata = {
            "output_image_id": output_image_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_params": params.dict(),
            "status": "failed",
            "error_message": str(e)
        }

    metadata_save_path = os.path.join(METADATA_DIR, f"{output_image_id}.json")
    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_save_path}")


# --- Endpoints ---
@app.post("/upload_face_images/", summary="Upload 1 to 5 face images")
async def api_upload_face_images(files: List[UploadFile] = File(..., max_items=5)):
    global uploaded_file_paths
    if not 1 <= len(files) <= 5: raise HTTPException(status_code=400, detail="1-5 face images.")
    saved_files_info = []
    for file in files:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail=f"Invalid type: {file.filename}. JPG/PNG only.")
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1].lower()
        save_filename = f"{file_id}{file_extension}"
        save_path = os.path.join(UPLOAD_DIR, save_filename)
        await save_upload_file(file, save_path)
        uploaded_file_paths[file_id] = save_path # Store full path
        saved_files_info.append({"id": file_id, "filename": file.filename, "stored_as": save_filename})
    return JSONResponse(content={"message": "Face images uploaded.", "files": saved_files_info})

@app.post("/upload_style_image/", summary="Upload style image")
async def api_upload_style_image(file: UploadFile = File(...)):
    global uploaded_file_paths
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid type. JPG/PNG only.")
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1].lower()
    save_filename = f"{file_id}{file_extension}"
    save_path = os.path.join(UPLOAD_DIR, save_filename)
    await save_upload_file(file, save_path)
    uploaded_file_paths[file_id] = save_path
    return JSONResponse(content={"message": "Style image uploaded.", "file": {"id": file_id, "filename": file.filename, "stored_as": save_filename}})

@app.post("/upload_pose_image/", summary="Upload pose image (optional)")
async def api_upload_pose_image(file: UploadFile = File(...)):
    global uploaded_file_paths
    if file.content_type != "image/png":
        raise HTTPException(status_code=400, detail="Invalid type. PNG only for pose.")
    file_id = str(uuid.uuid4())
    save_filename = f"{file_id}.png" # Standardize pose to .png
    save_path = os.path.join(UPLOAD_DIR, save_filename)
    await save_upload_file(file, save_path)
    uploaded_file_paths[file_id] = save_path
    return JSONResponse(content={"message": "Pose image uploaded.", "file": {"id": file_id, "filename": file.filename, "stored_as": save_filename}})

@app.post("/generate_image/", summary="Trigger image generation (async)")
async def api_generate_image(params: GenerateImageParams, background_tasks: BackgroundTasks):
    global uploaded_file_paths

    resolved_face_paths = []
    for fid in params.face_image_ids:
        path = uploaded_file_paths.get(fid)
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Uploaded face image with ID {fid} not found.")
        resolved_face_paths.append(path)

    style_image_path_resolved = uploaded_file_paths.get(params.style_image_id)
    if not style_image_path_resolved or not os.path.exists(style_image_path_resolved):
        raise HTTPException(status_code=404, detail=f"Uploaded style image ID {params.style_image_id} not found.")

    pose_image_path_resolved = None
    if params.pose_image_id:
        pose_image_path_resolved = uploaded_file_paths.get(params.pose_image_id)
        if not pose_image_path_resolved or not os.path.exists(pose_image_path_resolved):
            raise HTTPException(status_code=404, detail=f"Uploaded pose image ID {params.pose_image_id} not found.")

    output_image_id = str(uuid.uuid4())

    background_tasks.add_task(
        generate_image_task,
        output_image_id,
        resolved_face_paths,
        style_image_path_resolved,
        pose_image_path_resolved,
        params
    )

    return JSONResponse(content={
        "message": "Image generation task started.",
        "output_image_id": output_image_id,
        "retrieve_image_url": f"/get_image/{output_image_id}",
        "retrieve_metadata_url": f"/get_metadata/{output_image_id}"
    })

@app.get("/get_image/{image_id}", summary="Get generated image by ID")
async def api_get_image(image_id: str):
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
    if not os.path.exists(image_path):
        # Check if metadata exists and says "failed"
        metadata_path = os.path.join(METADATA_DIR, f"{image_id}.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f_meta:
                    meta = json.load(f_meta)
                    if meta.get("status") == "failed":
                        raise HTTPException(status_code=500, detail=f"Image generation failed. Error: {meta.get('error_message', 'Unknown error')}")
                    else: # In progress or other state
                        raise HTTPException(status_code=404, detail=f"Image not ready or not found. Status: {meta.get('status', 'Unknown')}")
            except Exception: # Fallback if metadata is malformed or other issue
                 raise HTTPException(status_code=404, detail="Image not found and metadata unreadable.")
        else: # No image, no metadata
            raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(image_path, media_type="image/png")

@app.get("/get_metadata/{image_id}", summary="Get image metadata by ID")
async def api_get_metadata(image_id: str):
    metadata_path = os.path.join(METADATA_DIR, f"{image_id}.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Metadata not found.")
    return FileResponse(metadata_path, media_type="application/json")


@app.get("/download_output_zip/{image_id}", summary="Download generated image and metadata as ZIP")
async def api_download_output_zip(image_id: str):
    image_filename = f"{image_id}.png"
    metadata_filename = f"{image_id}.json"

    image_path = os.path.join(IMAGE_DIR, image_filename)
    metadata_path = os.path.join(METADATA_DIR, metadata_filename)

    if not os.path.exists(image_path) or not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Output image or metadata not found.")

    zip_io = io.BytesIO()
    # Use ZIP_DEFLATED for compression if available, otherwise ZIP_STORED
    # Corrected: is_available('zlib')
    compression_method = zipfile.ZIP_DEFLATED if zipfile.is_available('zlib') else zipfile.ZIP_STORED

    with zipfile.ZipFile(zip_io, mode='w', compression=compression_method) as temp_zip:
        temp_zip.write(image_path, arcname=image_filename)
        temp_zip.write(metadata_path, arcname=metadata_filename)

    zip_io.seek(0) # Go to the beginning of the BytesIO buffer

    return StreamingResponse(
        zip_io,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=output_{image_id}.zip"}
    )

if __name__ == "__main__":
    import uvicorn
    # Note: Run with `uvicorn app.main:app --reload` from project root for development.
    # The load_models_on_startup will run when uvicorn starts the app.
    print(f"Starting FastAPI server. Access at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

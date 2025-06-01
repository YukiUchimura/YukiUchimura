from PIL import Image
import cv2
import torch
from diffusers import StableDiffusionXLAdapterPipeline, ControlNetModel, EulerDiscreteScheduler
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
import os
import numpy as np
from typing import Optional # Added for type hinting

# --- Global Variables & Model Cache ---
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models_hf")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

face_analysis_app = None
controlnet_instant_id = None
controlnet_style_shuffle = None
controlnet_openpose = None # New ControlNet for pose
sdxl_pipe = None

# --- Model Configuration ---
SDXL_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
INSTANTID_CONTROLNET_REPO_ID = "InstantX/InstantID"
STYLE_SHUFFLE_CONTROLNET_ID = "thibaud/controlnet-sdxl-shuffle"
OPENPOSE_CONTROLNET_ID = "thibaud/controlnet-openpose-sdxl-1.0" # SDXL OpenPose model

def load_models():
    global face_analysis_app, controlnet_instant_id, controlnet_style_shuffle, controlnet_openpose, sdxl_pipe

    if sdxl_pipe is not None:
        print("Models appear to be already loaded.")
        return

    print(f"Loading models from/to {MODEL_CACHE_DIR}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # 1. Face Analysis model
    try:
        print("Loading FaceAnalysis model (antelopev2)...")
        face_analysis_app = FaceAnalysis(name="antelopev2", root=MODEL_CACHE_DIR, providers=['CUDAExecutionProvider' if device == "cuda" else 'CPUExecutionProvider'])
        face_analysis_app.prepare(ctx_id=0, det_size=(640, 640))
        print("FaceAnalysis model loaded.")
    except Exception as e: print(f"Error loading FaceAnalysis model: {e}."); raise

    from huggingface_hub import hf_hub_download

    # 2. InstantID ControlNet
    try:
        print(f"Loading InstantID ControlNet model from {INSTANTID_CONTROLNET_REPO_ID}...")
        controlnet_instant_id_path = os.path.join(MODEL_CACHE_DIR, "instantid_control_model")
        os.makedirs(controlnet_instant_id_path, exist_ok=True)
        downloaded_instantid_file = hf_hub_download(
            repo_id=INSTANTID_CONTROLNET_REPO_ID, filename="control.safetensors",
            cache_dir=os.path.join(MODEL_CACHE_DIR, ".cache"), local_dir=controlnet_instant_id_path, local_dir_use_symlinks=False
        )
        controlnet_instant_id = ControlNetModel.from_single_file(downloaded_instantid_file, torch_dtype=torch_dtype)
        print("InstantID ControlNet model loaded.")
    except Exception as e: print(f"Error loading InstantID ControlNet: {e}"); raise

    # 3. Style Shuffle ControlNet
    try:
        print(f"Loading Style Shuffle ControlNet model ({STYLE_SHUFFLE_CONTROLNET_ID})...")
        controlnet_style_shuffle = ControlNetModel.from_pretrained(
            STYLE_SHUFFLE_CONTROLNET_ID, torch_dtype=torch_dtype, cache_dir=MODEL_CACHE_DIR
        )
        print("Style Shuffle ControlNet model loaded.")
    except Exception as e: print(f"Error loading Style Shuffle ControlNet: {e}"); raise

    # 4. OpenPose ControlNet (New)
    try:
        print(f"Loading OpenPose ControlNet model ({OPENPOSE_CONTROLNET_ID})...")
        controlnet_openpose = ControlNetModel.from_pretrained(
            OPENPOSE_CONTROLNET_ID, torch_dtype=torch_dtype, cache_dir=MODEL_CACHE_DIR
        )
        print("OpenPose ControlNet model loaded.")
    except Exception as e: print(f"Error loading OpenPose ControlNet: {e}"); raise

    # 5. SDXL Base Model + Scheduler + IP-Adapter for InstantID
    try:
        print(f"Loading SDXL base model ({SDXL_BASE_MODEL_ID}) and IP-Adapter...")
        all_controlnets = [controlnet_instant_id, controlnet_style_shuffle, controlnet_openpose]

        sdxl_pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            SDXL_BASE_MODEL_ID,
            controlnet=all_controlnets,
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None,
            use_safetensors=True, cache_dir=MODEL_CACHE_DIR,
        )

        ip_adapter_weights_path = hf_hub_download(
            repo_id=INSTANTID_CONTROLNET_REPO_ID, filename="ip-adapter.bin",
            cache_dir=os.path.join(MODEL_CACHE_DIR, ".cache"),
            local_dir=os.path.join(MODEL_CACHE_DIR, "instantid_ip_adapter"), local_dir_use_symlinks=False
        )
        sdxl_pipe.load_ip_adapter_instant_id(ip_adapter_weights_path)
        print("Loaded InstantID IP-Adapter weights.")

        sdxl_pipe.scheduler = EulerDiscreteScheduler.from_config(sdxl_pipe.scheduler.config)
        sdxl_pipe = sdxl_pipe.to(device)
        print("SDXL pipeline with all ControlNets and IP-Adapter ready.")
    except Exception as e: print(f"Error loading SDXL pipeline: {e}"); raise

    print("All models loaded successfully.")

def get_face_embeddings_and_control_images(face_image_paths: list[str]):
    global face_analysis_app
    if face_analysis_app is None:
        print("FaceAnalysis app not loaded. Attempting to load models first.")
        load_models()
    if not face_image_paths: raise ValueError("No face image paths provided.")
    face_pils, face_embeddings_list = [], []
    for image_path in face_image_paths:
        if not os.path.exists(image_path): print(f"Warning: Face image path not found: {image_path}"); continue
        pil_image = load_image(image_path); face_pils.append(pil_image)
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        faces = face_analysis_app.get(cv2_image)
        if not faces: print(f"Warning: No faces detected in {image_path}"); continue
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        face_embeddings_list.append(torch.tensor(faces[0].embedding).unsqueeze(0))
    if not face_embeddings_list: raise ValueError("Could not extract any face embeddings.")
    avg_face_embedding = torch.mean(torch.cat(face_embeddings_list, dim=0), dim=0, keepdim=True) if len(face_embeddings_list) > 1 else face_embeddings_list[0]
    control_image_face = face_pils[0] if face_pils else None
    print(f"Prepared face embedding. Using {face_image_paths[0] if control_image_face else 'None'} as ID control image.")
    return avg_face_embedding, control_image_face


def generate_image_with_instantid(
    face_image_paths: list[str],
    style_image_path: str,
    prompt: str,
    pose_image_path: Optional[str] = None,
    negative_prompt: str = "deformed, ugly, disfigured, low quality, blurry, nsfw, monochrome, grayscale, text, watermark, signature, extra fingers, mutated hands, bad anatomy",
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    instantid_scale: float = 0.8,
    style_shuffle_scale: float = 0.7,
    pose_control_scale: float = 0.75,
    ip_adapter_scale: float = 0.8
   ):
    global sdxl_pipe, controlnet_instant_id, controlnet_style_shuffle, controlnet_openpose
    if sdxl_pipe is None:
        print("Models not loaded. Attempting to load now...")
        load_models()
        if sdxl_pipe is None: raise RuntimeError("Models failed to load.")

    if not face_image_paths: raise ValueError("Face image paths are required.")
    if not os.path.exists(style_image_path): raise ValueError(f"Style image not found: {style_image_path}")

    face_embeddings, control_image_face = get_face_embeddings_and_control_images(face_image_paths)
    if control_image_face is None or face_embeddings is None:
        raise ValueError("Could not get valid control image or face embeddings for InstantID.")

    style_image_pil = load_image(style_image_path)
    device = sdxl_pipe.device
    face_embeddings = face_embeddings.to(device, dtype=sdxl_pipe.torch_dtype)
    sdxl_pipe.set_ip_adapter_scale(ip_adapter_scale)

    # Initialize lists for final control images and scales, matching the order of ControlNets at pipeline init
    # Order: [InstantID, Style Shuffle, OpenPose]
    final_control_images = [None, None, None]
    final_control_scales = [0.0, 0.0, 0.0]

    # 1. InstantID (always present)
    if controlnet_instant_id:
        final_control_images[0] = control_image_face
        final_control_scales[0] = instantid_scale
    else:
        print("Warning: InstantID ControlNet not available.")

    # 2. Style Shuffle (always present)
    if controlnet_style_shuffle:
        final_control_images[1] = style_image_pil
        final_control_scales[1] = style_shuffle_scale
    else:
        print("Warning: Style Shuffle ControlNet not available.")

    # 3. OpenPose (optional)
    if pose_image_path and os.path.exists(pose_image_path) and controlnet_openpose:
        print(f"Using pose image: {pose_image_path}")
        pose_image_pil = load_image(pose_image_path)
        final_control_images[2] = pose_image_pil
        final_control_scales[2] = pose_control_scale
    elif pose_image_path:
        print(f"Warning: Pose image path provided ({pose_image_path}) but image not found or OpenPose ControlNet not loaded.")

    active_cn_count = sum(1 for img in final_control_images if img is not None)
    print(f"Generating image with {active_cn_count} active ControlNets (InstantID, Style, Pose [optional])...")
    print(f"  Scales: ID={final_control_scales[0]}, Style={final_control_scales[1]}, Pose={final_control_scales[2]}")

    try:
        output_image = sdxl_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=final_control_images, # List of 3 images, some can be None
            ip_adapter_image_embeds=face_embeddings,
            controlnet_conditioning_scale=final_control_scales, # List of 3 scales
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
    except Exception as e:
        print(f"Error during image generation pipeline: {e}")
        import traceback; traceback.print_exc()
        output_image = Image.new('RGB', (512, 512), color = 'yellow')
        output_image.save("error_pose_generation_fallback.png")

    print("Image generation with pose control (if active) completed.")
    return output_image


if __name__ == '__main__':
    print("Running pipeline.py directly for testing InstantID + Style + Pose...")
    try:
        load_models()

        sample_files_dir = "sample_files_for_pipeline_main_pose"
        os.makedirs(sample_files_dir, exist_ok=True)
        sample_face_path = os.path.join(sample_files_dir, "sample_face.jpg")
        if not os.path.exists(sample_face_path):
            from urllib import request
            try: request.urlretrieve("https://raw.githubusercontent.com/InstantX/InstantID/main/assets/examples/yann-lecun.jpg", sample_face_path)
            except Exception as e: print(f"Face DL failed: {e}")

        sample_style_path = os.path.join(sample_files_dir, "sample_style.jpg")
        if not os.path.exists(sample_style_path):
            from urllib import request
            try: request.urlretrieve("https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sdxl/controlnet-shuffle/input_image_vermeer.png", sample_style_path)
            except Exception as e: print(f"Style DL failed: {e}")

        sample_pose_path = os.path.join(sample_files_dir, "sample_openpose.png")
        if not os.path.exists(sample_pose_path):
            from urllib import request
            try:
                print(f"Downloading sample OpenPose image to {sample_pose_path}...")
                request.urlretrieve("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/openpose.png", sample_pose_path)
            except Exception as e: print(f"Pose DL failed: {e}")

        if not all(os.path.exists(p) for p in [sample_face_path, sample_style_path, sample_pose_path]):
            print("Missing one or more sample files for full test. Aborting.")
            # return # Comment out to allow testing even if some downloads fail, e.g. no pose

        face_paths = [sample_face_path] if os.path.exists(sample_face_path) else []
        style_path = sample_style_path if os.path.exists(sample_style_path) else None
        pose_path_for_test = sample_pose_path if os.path.exists(sample_pose_path) else None

        if not face_paths or not style_path:
            print("Cannot run tests without face and style images.")
            return

        test_prompt = "A photo of a person in the specified pose and style, high quality, masterpiece"

        # Test 1: With Pose Control (if pose image available)
        if pose_path_for_test:
            print("\n--- Test 1: Generating with Pose Control ---")
            generated_img_with_pose = generate_image_with_instantid(
                face_image_paths=face_paths, style_image_path=style_path, prompt=test_prompt,
                pose_image_path=pose_path_for_test,
                num_inference_steps=30, instantid_scale=0.8, style_shuffle_scale=0.7, pose_control_scale=0.8, ip_adapter_scale=0.8
            )
            if generated_img_with_pose:
                generated_img_with_pose.save("test_output_with_pose.png")
                print(f"Saved test output with pose to test_output_with_pose.png")
        else:
            print("\n--- Test 1: Skipped (Pose image not available) ---")

        # Test 2: Without Pose Control
        print("\n--- Test 2: Generating without Pose Control ---")
        generated_img_no_pose = generate_image_with_instantid(
            face_image_paths=face_paths, style_image_path=style_path, prompt=test_prompt,
            pose_image_path=None,
            num_inference_steps=30, instantid_scale=0.8, style_shuffle_scale=0.7, pose_control_scale=0.8, ip_adapter_scale=0.8
        )
        if generated_img_no_pose:
            generated_img_no_pose.save("test_output_no_pose.png")
            print(f"Saved test output without pose to test_output_no_pose.png")

    except Exception as e:
        print(f"An error occurred during the __main__ pose test run: {e}")
        import traceback; traceback.print_exc()
    finally:
        print("Pose test run finished.")

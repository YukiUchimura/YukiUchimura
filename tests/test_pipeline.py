# tests/test_pipeline.py
import pytest
from app.image_generator.pipeline import (
    load_models,
    get_face_embeddings_and_control_images,
    generate_image_with_instantid
)
from PIL import Image
import os
import torch # For checking tensor types if needed
import numpy as np # For creating dummy cv2 image if needed by mocks

# Mock heavyweight model loading for most tests
@pytest.fixture(autouse=True) # Apply to all tests in this module
def mock_heavy_pipeline_components(mocker):
    # Mock FaceAnalysis
    mock_face_analysis_instance = mocker.MagicMock()
    mock_face_info = mocker.MagicMock()
    mock_face_info.embedding = np.array([0.1] * 512) # insightface returns numpy array
    mock_face_analysis_instance.get.return_value = [mock_face_info] # .get returns a list of faces
    mock_face_analysis_instance.prepare.return_value = None # .prepare usually returns None
    mocker.patch("app.image_generator.pipeline.FaceAnalysis", return_value=mock_face_analysis_instance)

    # Mock ControlNetModel loading
    mock_controlnet_model_instance = mocker.MagicMock()
    mocker.patch("app.image_generator.pipeline.ControlNetModel.from_single_file", return_value=mock_controlnet_model_instance)
    mocker.patch("app.image_generator.pipeline.ControlNetModel.from_pretrained", return_value=mock_controlnet_model_instance)

    # Mock SDXLAdapterPipeline
    mock_sdxl_pipe_instance = mocker.MagicMock()
    mock_sdxl_pipe_instance.device = "cpu"
    mock_sdxl_pipe_instance.torch_dtype = torch.float32
    dummy_pil_image = Image.new('RGB', (64, 64), color = 'blue')
    # The pipeline call returns an object that has an 'images' attribute, which is a list of PIL Images.
    mock_pipeline_output = mocker.MagicMock()
    mock_pipeline_output.images = [dummy_pil_image]
    mock_sdxl_pipe_instance.return_value = mock_pipeline_output # For when the pipeline instance is called
    # Also mock methods if called directly on the instance, e.g. load_ip_adapter_instant_id
    mock_sdxl_pipe_instance.load_ip_adapter_instant_id = mocker.MagicMock()
    mock_sdxl_pipe_instance.set_ip_adapter_scale = mocker.MagicMock()

    mocker.patch("app.image_generator.pipeline.StableDiffusionXLAdapterPipeline.from_pretrained", return_value=mock_sdxl_pipe_instance)

    # Mock hf_hub_download
    mocker.patch("app.image_generator.pipeline.hf_hub_download", return_value="mock_downloaded_path/file.safetensors")

    # Reset global model variables in the pipeline module before each test
    # to ensure load_models() attempts to load them (and thus use mocks)
    mocker.patch("app.image_generator.pipeline.face_analysis_app", new=None)
    mocker.patch("app.image_generator.pipeline.controlnet_instant_id", new=None)
    mocker.patch("app.image_generator.pipeline.controlnet_style_shuffle", new=None)
    mocker.patch("app.image_generator.pipeline.controlnet_openpose", new=None)
    mocker.patch("app.image_generator.pipeline.sdxl_pipe", new=None)


def test_load_models_mocked(mocker):
    # This test uses the autouse mock_heavy_pipeline_components fixture
    # We can assert that the global variables in pipeline.py are set to the mocks
    # by checking one of them.
    import app.image_generator.pipeline as pipeline_module

    pipeline_module.load_models()

    assert pipeline_module.face_analysis_app is not None
    assert pipeline_module.sdxl_pipe is not None
    # Check if a sub-mock (like prepare on FaceAnalysis) was called
    pipeline_module.face_analysis_app.prepare.assert_called_once()
    # Check if from_pretrained on SDXL pipeline was called
    pipeline_module.StableDiffusionXLAdapterPipeline.from_pretrained.assert_called()


def test_get_face_embeddings_mocked(tmp_path, mocker):
    # Ensure models are "loaded" (i.e., mocks are in place via autouse fixture)
    # This will call load_models() internally if pipeline.face_analysis_app is None
    import app.image_generator.pipeline as pipeline_module
    if pipeline_module.face_analysis_app is None: # If fixture didn't set it through load_models
        pipeline_module.load_models()


    dummy_face_path = tmp_path / "face.jpg"
    Image.new('RGB', (64, 64), color = 'red').save(dummy_face_path)

    avg_embedding, control_image = get_face_embeddings_and_control_images([str(dummy_face_path)])

    assert avg_embedding is not None
    assert isinstance(avg_embedding, torch.Tensor)
    assert avg_embedding.shape == (1, 512) # (1, embedding_dim) after unsqueeze and potential mean
    assert control_image is not None
    assert isinstance(control_image, Image.Image)
    pipeline_module.face_analysis_app.get.assert_called() # Verify insightface was used


def test_generate_image_with_instantid_mocked_basic_call(tmp_path, mocker):
    import app.image_generator.pipeline as pipeline_module
    # Ensure models are loaded (mocks will be used)
    pipeline_module.load_models()

    face_path = tmp_path / "face.png"
    style_path = tmp_path / "style.png"
    Image.new('RGB', (64,64), color='red').save(face_path)
    Image.new('RGB', (64,64), color='blue').save(style_path)

    img_result = generate_image_with_instantid(
        face_image_paths=[str(face_path)],
        style_image_path=str(style_path),
        prompt="A test image",
    )
    assert isinstance(img_result, Image.Image)
    assert img_result.size == (64,64) # Check if dummy image from mock is returned
    # Assert that the mocked pipeline was called
    pipeline_module.sdxl_pipe.assert_called_once()

def test_generate_image_with_pose_mocked(tmp_path, mocker):
    import app.image_generator.pipeline as pipeline_module
    pipeline_module.load_models() # Ensure all mocks are set up

    face_path = tmp_path / "face.png"
    style_path = tmp_path / "style.png"
    pose_path = tmp_path / "pose.png"
    Image.new('RGB', (64,64), color='red').save(face_path)
    Image.new('RGB', (64,64), color='blue').save(style_path)
    Image.new('RGB', (64,64), color='green').save(pose_path)

    img_result = generate_image_with_instantid(
        face_image_paths=[str(face_path)],
        style_image_path=str(style_path),
        prompt="A test image with pose",
        pose_image_path=str(pose_path)
    )
    assert isinstance(img_result, Image.Image)

    # Check that the pipeline was called, and inspect arguments if needed
    pipeline_module.sdxl_pipe.assert_called_once()
    args, kwargs = pipeline_module.sdxl_pipe.call_args

    # final_control_images is passed as 'image' keyword argument
    # Expected order: [ID_img, Style_img, Pose_img]
    assert kwargs['image'][2] is not None # Check that pose image was passed
    assert kwargs['controlnet_conditioning_scale'][2] > 0.0 # Check pose scale is active

def test_generate_image_without_pose_mocked(tmp_path, mocker):
    import app.image_generator.pipeline as pipeline_module
    pipeline_module.load_models()

    face_path = tmp_path / "face.png"
    style_path = tmp_path / "style.png"
    Image.new('RGB', (64,64), color='red').save(face_path)
    Image.new('RGB', (64,64), color='blue').save(style_path)

    img_result = generate_image_with_instantid(
        face_image_paths=[str(face_path)],
        style_image_path=str(style_path),
        prompt="A test image without pose",
        pose_image_path=None # Explicitly no pose
    )
    assert isinstance(img_result, Image.Image)
    pipeline_module.sdxl_pipe.assert_called_once()
    args, kwargs = pipeline_module.sdxl_pipe.call_args
    assert kwargs['image'][2] is None # Check that pose image is None
    assert kwargs['controlnet_conditioning_scale'][2] == 0.0 # Check pose scale is inactive

# TODO: Add tests for specific error conditions, like no faces found.
# This would involve changing the return value of mock_face_app.get([])
# and asserting that get_face_embeddings_and_control_images raises ValueError.

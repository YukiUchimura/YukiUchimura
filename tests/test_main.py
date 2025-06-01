# tests/test_main.py
import pytest
from httpx import AsyncClient
from fastapi import status # For status codes
import os
import shutil
from app.main import app, UPLOAD_DIR, IMAGE_DIR, METADATA_DIR, uploaded_file_paths, GenerateImageParams # Import your app and config

# Helper to clean up a specific file ID from the mock registry and filesystem
def cleanup_uploaded_file(file_id):
    if file_id in uploaded_file_paths:
        file_path_to_remove = uploaded_file_paths[file_id]
        if os.path.exists(file_path_to_remove):
            os.remove(file_path_to_remove)
        del uploaded_file_paths[file_id]

@pytest.fixture(autouse=True)
def clear_global_uploaded_paths_and_dirs():
    # Clear before test
    uploaded_file_paths.clear()
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR) # Remove temp upload dir
    os.makedirs(UPLOAD_DIR, exist_ok=True) # Recreate it

    yield # Test runs here

    # Clear after test
    uploaded_file_paths.clear()
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists(IMAGE_DIR): # Clean output dirs if tests create files there
        shutil.rmtree(IMAGE_DIR)
    if os.path.exists(METADATA_DIR):
        shutil.rmtree(METADATA_DIR)
    # Recreate them for subsequent app loads or other tests not in this module
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)


@pytest.mark.asyncio
async def test_read_docs(test_client: AsyncClient):
    response = await test_client.get("/docs")
    assert response.status_code == status.HTTP_200_OK

@pytest.mark.asyncio
async def test_upload_face_image_success(test_client: AsyncClient, tmp_path):
    dummy_image_content = b"dummy image data"
    file_name = "face.jpg"
    file_path = tmp_path / file_name
    file_path.write_bytes(dummy_image_content)

    with open(file_path, "rb") as f:
        response = await test_client.post("/upload_face_images/", files={"files": (file_name, f, "image/jpeg")})

    assert response.status_code == status.HTTP_200_OK
    json_response = response.json()
    assert "files" in json_response
    assert len(json_response["files"]) == 1
    assert "id" in json_response["files"][0]
    file_id = json_response["files"][0]["id"]
    assert file_id in uploaded_file_paths
    assert os.path.exists(uploaded_file_paths[file_id])
    # Autouse fixture will clean up


@pytest.mark.asyncio
async def test_upload_style_image_invalid_type(test_client: AsyncClient, tmp_path):
    dummy_text_content = b"this is not an image"
    file_name = "style.txt"
    file_path = tmp_path / file_name
    file_path.write_bytes(dummy_text_content)

    with open(file_path, "rb") as f:
        response = await test_client.post("/upload_style_image/", files={"file": (file_name, f, "text/plain")})

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid file type" in response.json()["detail"]

@pytest.mark.asyncio
async def test_generate_image_missing_face_id(test_client: AsyncClient, mocker):
    mocker.patch("app.main.generate_image_task") # Mock the actual background task

    # Prepare a valid style ID and its dummy file
    style_id = "valid_style_id_for_face_test"
    dummy_style_file_path = os.path.join(UPLOAD_DIR, f"{style_id}.jpg")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(dummy_style_file_path, "wb") as f: f.write(b"dummy style content")
    uploaded_file_paths[style_id] = dummy_style_file_path

    # Construct payload using defaults from Pydantic model where possible
    default_params = GenerateImageParams(face_image_ids=[], style_image_id="").dict()
    payload = {
        **default_params,
        "face_image_ids": ["non_existent_face_id"],
        "style_image_id": style_id,
        "prompt": "test prompt for missing face"
    }

    response = await test_client.post("/generate_image/", json=payload)

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Uploaded face image with ID non_existent_face_id not found" in response.json()["detail"]
    # Autouse fixture will clean up


@pytest.mark.asyncio
async def test_generate_image_success_mocked(test_client: AsyncClient, mocker, tmp_path):
    mock_add_task = mocker.MagicMock()
    mocker.patch("fastapi.BackgroundTasks.add_task", new=mock_add_task)

    face_id = "test_face_id_gen_success"
    style_id = "test_style_id_gen_success"

    # Create dummy files in UPLOAD_DIR as the app expects them there based on uploaded_file_paths registry
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    face_file_on_server = os.path.join(UPLOAD_DIR, f"{face_id}.jpg")
    style_file_on_server = os.path.join(UPLOAD_DIR, f"{style_id}.jpg")

    with open(face_file_on_server, "wb") as f: f.write(b"dummyface")
    with open(style_file_on_server, "wb") as f: f.write(b"dummystyle")

    uploaded_file_paths[face_id] = face_file_on_server
    uploaded_file_paths[style_id] = style_file_on_server

    default_params = GenerateImageParams(face_image_ids=[face_id], style_image_id=style_id).dict()
    payload = {
        **default_params,
        "prompt": "A photo of a person",
        # Other params will use defaults from Pydantic model
    }
    response = await test_client.post("/generate_image/", json=payload)

    assert response.status_code == status.HTTP_200_OK
    json_response = response.json()
    assert "output_image_id" in json_response
    mock_add_task.assert_called_once()
    # Autouse fixture will clean up the dummy files and registry

# TODO: Add tests for /get_image and /get_metadata
# These would involve mocking the generate_image_task to create dummy output files
# or using a known output_image_id from a (mocked) successful generation.

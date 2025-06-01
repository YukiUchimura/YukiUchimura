# SDXL InstantID Generator

This project implements an image generation workflow using Stable Diffusion XL (SDXL) and InstantID. It allows users to generate images of a specific person (from reference face images) in a particular style (from a style image) and optionally with a specific pose (from an OpenPose image). The application features a web UI for easy interaction and is containerized using Docker for deployment.

## Features

*   **Identity Preservation:** Uses InstantID to maintain the identity from 1-5 reference face images.
*   **Style Transfer:** Applies the artistic style (colors, textures, atmosphere) from a provided style image using ControlNet Shuffle.
*   **Pose Control:** Optionally guides the generated image's pose using an OpenPose reference image.
*   **Web Interface:** User-friendly UI built with React for uploading images, setting parameters, and viewing results.
*   **Async Backend:** FastAPI backend handles image generation as a background task.
*   **Dockerized:** Fully containerized for consistent setup and deployment.
*   **CI/CD:** Includes GitHub Actions for automated unit testing.
*   **Metadata & Download:** Saves generation metadata and allows downloading of images, metadata, and a combined ZIP archive.

## Prerequisites

*   **Git:** For cloning the repository.
*   **Docker & Docker Compose:** To build and run the application. (Ensure Docker Compose v2 `docker compose` or standalone `docker-compose` is installed).
*   **NVIDIA GPU & Drivers (Recommended for GPU acceleration):**
    *   NVIDIA GPU with at least 11GB VRAM (e.g., GTX 1080Ti or newer) for optimal performance with 1024x1024 images.
    *   Up-to-date NVIDIA drivers.
    *   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed to enable GPU access within Docker containers.
*   **Node.js and npm/yarn (Optional):** Only if you want to run the React frontend in development mode separately or modify frontend code.

## Setup & Running

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd sdxl-instantid-generator
    ```
    *(Replace `<repository_url>` with the actual URL of this repository)*

2.  **Using the Setup Script (Linux/macOS):**
    The `setup.sh` script will guide you through building and starting the application.
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```
    This will:
    *   Check for Docker and Docker Compose.
    *   Create necessary local directories (`./models_hf`, `./outputs`, `./uploads`).
    *   Build the Docker image using `docker compose build`.
    *   Start the application using `docker compose up`.

3.  **Manual Docker Compose (Alternative):**
    If you prefer to run commands manually:
    ```bash
    # Ensure these directories exist in the project root
    mkdir -p ./models_hf ./outputs ./uploads

    # Build the Docker image
    docker compose build

    # Start the application
    # This will download models on the first run if not present in ./models_hf
    # Ensure your Docker daemon is configured to use NVIDIA runtime for GPU.
    docker compose up
    ```
    To run in detached mode: `docker compose up -d`.
    To stop: `docker compose down`.

4.  **Accessing the Services:**
    *   **Backend API & UI (if served by FastAPI in future):** `http://localhost:7860`
    *   **API Docs (Swagger UI):** `http://localhost:7860/docs`
    *   **React Frontend (Development Mode - if run separately):**
        If you want to run the frontend dev server:
        ```bash
        cd app/frontend
        npm install # or yarn install
        npm start   # or yarn start
        ```
        The React development server will typically be available at `http://localhost:3000`. The React app is configured to make API calls to `http://localhost:7860`.

**Important Notes on First Run:**
*   The initial `docker compose build` might take some time.
*   When the application starts for the first time (inside the Docker container), it will download several gigabytes of machine learning models into the `./models_hf` directory. This can take a significant amount of time depending on your internet connection. Subsequent runs will be much faster as models will be cached.

## Usage

1.  Open your web browser and navigate to the React application (usually `http://localhost:3000` if run via `npm start`, or `http://localhost:7860` if eventually served by FastAPI).
2.  **Upload Images:**
    *   **Face Images:** Upload 1 to 5 clear photos of the target person's face (JPG/PNG).
    *   **Style Image:** Upload an image whose artistic style you want to emulate (JPG/PNG).
    *   **Pose Image (Optional):** Upload an OpenPose skeleton image (PNG) if you want to control the pose.
3.  **Adjust Parameters:**
    *   **Prompt:** Describe the desired scene or subject.
    *   **Face Scale (Identity):** Controls how strongly the identity from face images is applied (0.0 to 1.5).
    *   **Style Scale (Shuffle):** Controls the intensity of style transfer (0.0 to 1.0).
    *   **Steps:** Number of inference steps (e.g., 20-50). More steps can improve quality but take longer.
    *   **Seed:** Leave empty for a random seed, or enter a number for reproducible results (if the same seed is used with same inputs).
4.  **Generate:** Click the "Generate Image" button.
5.  **View & Download:**
    *   A status message will indicate progress.
    *   Once complete, the generated image will appear.
    *   You can download the image, its metadata (JSON), or a ZIP archive containing both.

## Testing

To run the unit tests (requires Python and `pytest` installed in your environment, or run inside a Docker container with test dependencies):
```bash
# Ensure test dependencies are installed (they are in requirements.txt)
# pip install pytest pytest-asyncio httpx pytest-mock

pytest tests/
```
The tests are also run automatically via GitHub Actions on pushes/PRs to main/develop branches.

## Project Structure

```
.
├── .github/workflows/        # GitHub Actions CI configuration
├── app/
│   ├── frontend/             # React frontend application (Node.js)
│   │   ├── public/
│   │   └── src/
│   ├── image_generator/      # Core Python image generation pipeline (SDXL, InstantID)
│   │   └── pipeline.py
│   ├── main.py               # FastAPI backend application
│   └── __init__.py
├── docker/
│   └── Dockerfile            # Dockerfile for the backend service
├── models_hf/                # (Created on run) Cached Hugging Face models
├── outputs/                  # (Created on run) Generated images and metadata
├── scripts/                  # Helper scripts (currently empty)
├── tests/                    # Pytest unit tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_main.py
│   └── test_pipeline.py
├── uploads/                  # (Created on run) Temporarily stores uploaded images
├── .dockerignore
├── docker-compose.yml        # Docker Compose configuration
├── LICENSE                   # (Should be added)
├── pytest.ini                # Pytest configuration
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── setup.sh                  # Setup script for Linux/macOS
```

## Technology Stack

*   **Backend:** Python, FastAPI, Uvicorn
*   **Image Generation:** Diffusers (Hugging Face), PyTorch, SDXL, InstantID, ControlNets (Shuffle, OpenPose), Insightface
*   **Frontend:** React, JavaScript, HTML, CSS
*   **Containerization:** Docker, Docker Compose
*   **Testing:** Pytest, HTTPX, pytest-mock
*   **CI:** GitHub Actions

## Future Enhancements / TODOs

*   Implement the "low-load generation test (256x256)" in the CI pipeline (F-06).
*   More robust face masking for the style image if it contains prominent faces.
*   Serve React frontend static build from FastAPI for a single deployment unit.
*   Add progress bar with more granular updates (e.g., using WebSockets or Server-Sent Events if backend supports step-wise progress).
*   Add a `setup.cmd` for Windows users.
*   Add a `LICENSE` file.
*   Explore further performance optimizations (e.g., ONNX runtime for models, TensorRT).
```

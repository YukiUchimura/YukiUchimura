# Start from an official Python base image.
# Using a specific version for better reproducibility.
# Consider a slim version if space becomes a major concern, but full might have more build tools.
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables
# - To prevent Python from writing pyc files to disc (optional, but can be good practice)
ENV PYTHONDONTWRITEBYTECODE 1
# - To prevent Python from buffering stdout and stderr (important for logging)
ENV PYTHONUNBUFFERED 1
# - Placeholder for where models might be cached within the container
ENV MODEL_CACHE_DIR /app/models_hf

# Install system dependencies that might be needed by Python libraries
# e.g., libgl1 for OpenCV, libsm6, libxext6
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This should be one of the last steps to leverage Docker build caching.
COPY ./app /app/app

# Create the model cache directory (if it's not created by model downloaders automatically)
RUN mkdir -p ${MODEL_CACHE_DIR}

# (Optional) Add a non-root user for security
# RUN useradd -ms /bin/bash appuser
# USER appuser

# Default command to run when the container starts (can be overridden).
# For now, just run a simple python command or leave it for docker-compose to define.
# CMD ["python", "app/image_generator/pipeline.py"]
# Default command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

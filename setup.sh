#!/bin/bash
# setup.sh - Basic setup and run script for SDXL InstantID Generator

echo "SDXL InstantID Generator Setup"
echo "--------------------------------"

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker could not be found. Please install Docker first."
    echo "Visit https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed (v2 syntax: docker compose)
if ! docker compose version &> /dev/null
then
    echo "Docker Compose (v2 syntax: 'docker compose') could not be found."
    echo "It's usually included with Docker Desktop. For Linux, see: https://docs.docker.com/compose/install/"
    # Attempt to check for older 'docker-compose' if 'docker compose' fails
    if ! command -v docker-compose &> /dev/null
    then
        echo "Older 'docker-compose' also not found. Please install Docker Compose."
        exit 1
    else
        echo "Found older 'docker-compose'. Will use that."
        COMPOSE_CMD="docker-compose"
    fi
else
    COMPOSE_CMD="docker compose"
fi

echo "Using Docker Compose command: $COMPOSE_CMD"

# Ensure directories for volumes exist (Docker typically creates them, but good practice)
echo "Ensuring local directories for Docker volumes exist..."
mkdir -p ./models_hf
mkdir -p ./outputs
mkdir -p ./uploads

echo ""
echo "Building Docker image (this might take a while on first run)..."
$COMPOSE_CMD build
if [ $? -ne 0 ]; then
    echo "Docker build failed. Please check error messages."
    exit 1
fi

echo ""
echo "Starting the application using Docker Compose..."
echo "FastAPI backend will be available on http://localhost:7860"
echo "React frontend (if run separately using npm start) on http://localhost:3000"
echo "Press Ctrl+C to stop."
echo ""

$COMPOSE_CMD up

# To run in detached mode, use:
# $COMPOSE_CMD up -d
# To stop, use:
# $COMPOSE_CMD down

echo "Application stopped."

#!/bin/bash

# Ensure the script fails on any error
set -e

# Default values
IMAGE_NAME="meta-review"
CONTAINER_NAME="meta-review-container"

# Check if a container with the same name already exists
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Removing existing container..."
    docker rm -f ${CONTAINER_NAME}
fi

# Run the container
docker run --gpus all \
    --name ${CONTAINER_NAME} \
    --shm-size=64g \
    -it \
    --rm \
    -v $(pwd):/app \
    -v $HOME/.cache:/root/.cache \
    ${IMAGE_NAME} "$@" 
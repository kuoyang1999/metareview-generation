# Use Miniconda3 as the base image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Copy the Conda environment file into the container
COPY environment.yml /app/

# Create the Conda environment
RUN conda env create -f /app/environment.yml

# Make sure the environment is activated by default
SHELL ["conda", "run", "-n", "longLoRA", "/bin/bash", "-c"]

# Copy the entire project into the container
COPY . /app/

# Expose any ports needed (e.g., for serving APIs)
EXPOSE 8000

# Set the default command to run the project (e.g., training script)
CMD ["python", "scripts/python/train.py"]
# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install basic utilities
RUN apt-get update && apt-get install -y \
    wget \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Verify CUDA installation
RUN nvcc --version

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Initialize conda in bash
RUN conda init bash && \
    echo "conda activate meta-review" >> ~/.bashrc

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set the working directory in the container
WORKDIR /app

# Add the project directory to PYTHONPATH
ENV PYTHONPATH=/app

# Copy the Conda environment file into the container
COPY environment.yml /app/

# Create the Conda environment and verify it exists
RUN conda env create -f environment.yml && \
    conda env list && \
    conda run -n meta-review python --version

# Copy the entire project into the container
COPY . /app/

# Expose any ports needed (e.g., for serving APIs)
EXPOSE 8000

# Set the default command to run the project
CMD ["conda", "run", "--no-capture-output", "-n", "meta-review", "torchrun", "--nproc_per_node=1", \
    "scripts/python/train.py", \
    "--max_samples", "1", \
    "--bf16", "True", \
    "--model_name_or_path", "meta-llama/Llama-2-7b-hf", \
    "--output_dir", "", \
    "--use_flash_attn", "True", \
    "--low_rank_training", "True", \
    "--gradient_accumulation_steps", "1", \
    "--num_train_epochs", "20", \
    "--per_device_train_batch_size", "1", \
    "--per_device_eval_batch_size", "1", \
    "--eval_strategy", "no", \
    "--save_strategy", "steps", \
    "--save_steps", "1000", \
    "--save_total_limit", "10", \
    "--learning_rate", "2e-5", \
    "--weight_decay", "0.0", \
    "--warmup_steps", "20", \
    "--lr_scheduler_type", "constant_with_warmup", \
    "--logging_steps", "1", \
    "--tf32", "True", \
    "--deepspeed", "configs/ds_configs/stage3.json"]
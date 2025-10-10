<<<<<<< HEAD
# Use NVIDIA CUDA runtime with cuDNN
FROM nvidia/cuda:12.1.105-cudnn8-runtime-ubuntu22.04

# Set environment variables to avoid Python buffering and warnings
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
=======
# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1-base-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
>>>>>>> 4906eee39bbe85c6561aca5f93b76d0d8e32bdce

# Copy the rest of the application
COPY . .

<<<<<<< HEAD
# Create volume for models or data
VOLUME ["/app/models"]

# Default command to run example.py
CMD ["python3.10", "example.py"]
=======
# Create volume for data
VOLUME ["/app/data"]

# Default command to run example
CMD ["python3", "example.py"]
>>>>>>> 4906eee39bbe85c6561aca5f93b76d0d8e32bdce

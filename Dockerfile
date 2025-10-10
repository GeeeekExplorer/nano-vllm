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

# Copy the rest of the application
COPY . .

# Create volume for data
VOLUME ["/app/data"]

# Default command to run example
CMD ["python3", "example.py"]
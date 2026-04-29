# Build for linux/amd64 so pip can fetch cu121 wheels (PyTorch has no aarch64 cu121 builds)
FROM --platform=linux/amd64 nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ARG PYTHON_VERSION=3

ENV PIP_BREAK_SYSTEM_PACKAGES=1


RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace
COPY . /workspace

RUN pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121

RUN pip install xxhash transformers accelerate safetensors tqdm huggingface_hub

CMD ["python", "example.py"]
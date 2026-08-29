FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive

# Known-good detector stack validated on Orin Nano SDK
ARG TENSORRT_VERSION=10.4.0.26
ARG TENSORRT_URL=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/tars/TensorRT-${TENSORRT_VERSION}.l4t.aarch64-gnu.cuda-12.6.tar.gz
ARG NANOOWL_REPO=https://github.com/NVIDIA-AI-IOT/nanoowl.git
ARG NANOOWL_COMMIT=fb553dee4c58f8c53ec2bf01f355a54648e67dbd
ARG TORCH2TRT_REPO=https://github.com/NVIDIA-AI-IOT/torch2trt.git
ARG TORCH2TRT_COMMIT=4e820ae31b4e35d59685935223b05b2e11d47b03

RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    git \
    curl \
    wget \
    pkg-config \
    gdb \
    strace \
    libopencv-dev \
    libv4l-dev \
    libyaml-cpp-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-opencv \
    python3-numpy \
    python3-setuptools \
    python3-wheel \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    v4l-utils \
    i2c-tools \
    zsh \
    && rm -rf /var/lib/apt/lists/*

RUN chsh -s /usr/bin/zsh root

RUN mkdir -p /tmp/tensorrt \
    && wget --quiet --show-progress \
        "${TENSORRT_URL}" \
        -O /tmp/tensorrt/TensorRT.tar.gz \
    && tar -xzf /tmp/tensorrt/TensorRT.tar.gz -C /usr/src \
    && mv /usr/src/TensorRT-* /usr/src/tensorrt \
    && cp -r /usr/src/tensorrt/lib/* /usr/lib/aarch64-linux-gnu/ \
    && cp -r /usr/src/tensorrt/include/* /usr/include/aarch64-linux-gnu/ \
    && PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')" \
    && pip3 install --no-cache-dir \
        /usr/src/tensorrt/python/tensorrt-*-cp${PY_VERSION}-*.whl \
    && ldconfig \
    && rm -rf /tmp/tensorrt

RUN pip3 install --no-cache-dir \
    torch==2.5.0 \
    torchvision==0.20.0 \
    transformers==4.48.3

RUN git clone "${TORCH2TRT_REPO}" /opt/torch2trt \
    && cd /opt/torch2trt \
    && git checkout "${TORCH2TRT_COMMIT}" \
    && pip3 install --no-cache-dir . \
    && cmake -B build -DCUDA_ARCHITECTURES=87 . \
    && cmake --build build --target install -j"$(nproc)" \
    && ldconfig

RUN git clone "${NANOOWL_REPO}" /opt/nanoowl \
    && cd /opt/nanoowl \
    && git checkout "${NANOOWL_COMMIT}" \
    && pip3 install --no-cache-dir -e .

RUN python3 - <<'PY'
import torch
import torchvision
import transformers
import tensorrt
import torch2trt
import nanoowl

assert torch.__version__.startswith("2.5.0")
assert torchvision.__version__.startswith("0.20.0")
assert transformers.__version__ == "4.48.3"
assert tensorrt.__version__ == "10.4.0"

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("TensorRT:", tensorrt.__version__)
print("torch2trt:", torch2trt.__file__)
print("NanoOWL:", nanoowl.__file__)
PY
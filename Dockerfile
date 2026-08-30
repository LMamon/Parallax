FROM dustynv/nanoowl:r36.4.0 AS nanoowl

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive

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
    nlohmann-json3-dev \
    libopencv-dev \
    libopenblas0 \
    libv4l-dev \
    libyaml-cpp-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgtest-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-opencv \
    python3-numpy \
    python3-setuptools \
    python3-wheel \
    v4l-utils \
    i2c-tools \
    zsh \
    && rm -rf /var/lib/apt/lists/*

RUN chsh -s /usr/bin/zsh root

# NanoOWL detector stack.
COPY --from=nanoowl /opt/nanoowl /opt/nanoowl
COPY --from=nanoowl /opt/torch2trt /opt/torch2trt

COPY --from=nanoowl \
    /usr/local/lib/python3.10/dist-packages \
    /usr/local/lib/python3.10/dist-packages

# TensorRT 10.4 userspace used by the detector.
COPY --from=nanoowl \
    /usr/lib/aarch64-linux-gnu/libnvinfer.so.10 \
    /usr/lib/aarch64-linux-gnu/libnvinfer.so.10

COPY --from=nanoowl \
    /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10 \
    /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10

COPY --from=nanoowl \
    /usr/lib/aarch64-linux-gnu/libnvinfer_lean.so.10 \
    /usr/lib/aarch64-linux-gnu/libnvinfer_lean.so.10

COPY --from=nanoowl \
    /usr/lib/aarch64-linux-gnu/libnvinfer_dispatch.so.10 \
    /usr/lib/aarch64-linux-gnu/libnvinfer_dispatch.so.10

COPY --from=nanoowl \
    /usr/lib/aarch64-linux-gnu/libnvinfer_vc_plugin.so.10 \
    /usr/lib/aarch64-linux-gnu/libnvinfer_vc_plugin.so.10

COPY --from=nanoowl \
    /usr/lib/aarch64-linux-gnu/do_not_link_against_nvinfer_builder_resource \
    /usr/lib/aarch64-linux-gnu/do_not_link_against_nvinfer_builder_resource

COPY --from=nanoowl \
    /usr/include/aarch64-linux-gnu/NvInfer.h \
    /usr/include/aarch64-linux-gnu/NvInfer.h

COPY --from=nanoowl /usr/src/tensorrt /opt/tensorrt-10.4

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/opt/tensorrt-10.4/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/workspace/Parallax/python:/opt/nanoowl:/opt/torch2trt

RUN ldconfig
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
    v4l-utils \
    i2c-tools \
    zsh \
    && rm -rf /var/lib/apt/lists/*

RUN chsh -s /usr/bin/zsh root

# pytorch
# tbd
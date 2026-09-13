FROM dustynv/nanoowl:r36.4.0 AS nanoowl

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive
ARG NVBLOX_VERSION=v0.0.10
ARG RPLIDAR_SDK_REF=master

RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    git \
    git-lfs \
    curl \
    wget \
    pkg-config \
    gdb \
    strace \
    nlohmann-json3-dev \
    libopencv-dev \
    libeigen3-dev \
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

# TensorRT 10.4 SDK used by perception backends.
COPY --from=nanoowl /usr/src/tensorrt /opt/tensorrt-10.4


RUN git clone --depth 1 \
        --branch "${RPLIDAR_SDK_REF}" \
        https://github.com/Slamtec/rplidar_sdk.git \
        /root/rplidar_sdk \
    && make -C /root/rplidar_sdk -j"$(nproc)"

# nvblox GPU mapping library.
# Pin the release rather than tracking the moving `public` branch.
RUN git clone --branch ${NVBLOX_VERSION} \
        --depth 1 \
        https://github.com/nvidia-isaac/nvblox.git \
        /opt/nvblox \
    && cmake -S /opt/nvblox -B /opt/nvblox/build \
        -DCMAKE_CUDA_ARCHITECTURES=87 \
        -DBUILD_PYTORCH_WRAPPER=OFF \
        -DBUILD_TESTING=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_RENDERER=OFF \
    && cmake --build /opt/nvblox/build -j4

ENV TENSORRT_ROOT=/opt/tensorrt-10.4 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/opt/tensorrt-10.4/targets/aarch64-linux-gnu/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONPATH=/workspace/Parallax/python:/opt/nanoowl:/opt/torch2trt

RUN ldconfig
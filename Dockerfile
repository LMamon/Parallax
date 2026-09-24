FROM dustynv/nanoowl:r36.4.0 AS nanoowl

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive
ARG NVBLOX_VERSION=v0.0.10
ARG RPLIDAR_SDK_REF=master

SHELL ["/bin/bash", "-c"]

# Keep the known-good Parallax toolchain while the ROS workspace replaces
# infrastructure incrementally instead of forcing a flag-day migration.
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    gdb \
    git \
    git-lfs \
    gnupg2 \
    i2c-tools \
    libeigen3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgtest-dev \
    libopenblas0 \
    libopencv-dev \
    libv4l-dev \
    libyaml-cpp-dev \
    lsb-release \
    nlohmann-json3-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    python3-opencv \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python3-wheel \
    software-properties-common \
    strace \
    v4l-utils \
    wget \
    zsh \
    && rm -rf /var/lib/apt/lists/*

RUN chsh -s /usr/bin/zsh root

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu \
        $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
        > /etc/apt/sources.list.d/ros2.list

# Isaac ROS 3.2 uses the release-3 apt channel on Jammy.
# Install ROS and the accelerated packages as binaries rather than rebuilding
# NVIDIA's stack inside Parallax.
RUN wget -qO - https://isaac.download.nvidia.com/isaac-ros/repos.key | apt-key add - \
    && echo "deb https://isaac.download.nvidia.com/isaac-ros/release-3 $(lsb_release -cs) release-3.0" \
       > /etc/apt/sources.list.d/isaac-ros.list \
    && apt-get update \
    && apt-get install -y \
        python3-colcon-common-extensions \
        python3-rosdep \
        ros-humble-ament-cmake \
        ros-humble-camera-info-manager \
        ros-humble-cv-bridge \
        ros-humble-foxglove-bridge \
        ros-humble-image-transport \
        ros-humble-isaac-ros-image-proc \
        ros-humble-isaac-ros-nvblox \
        ros-humble-isaac-ros-stereo-image-proc \
        ros-humble-isaac-ros-visual-slam \
        ros-humble-nav-msgs \
        ros-humble-ompl \
        ros-humble-ros-base \
        ros-humble-sensor-msgs \
        ros-humble-tf2-ros \
        ros-humble-tf2-tools \
        ros-humble-v4l2-camera \
        ros-humble-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# NanoOWL remains available in the same runtime. Its lifecycle integration is
# intentionally deferred until the spatial spine is alive.
COPY --from=nanoowl /opt/nanoowl /opt/nanoowl
COPY --from=nanoowl /opt/torch2trt /opt/torch2trt
COPY --from=nanoowl \
    /usr/local/lib/python3.10/dist-packages \
    /usr/local/lib/python3.10/dist-packages
COPY --from=nanoowl /usr/src/tensorrt /opt/tensorrt-10.4

# These two source builds are transitional compatibility for the inherited
# C++ test target. ROS-facing LiDAR and mapping will replace them in later gates.
RUN git clone --depth 1 \
        --branch "${RPLIDAR_SDK_REF}" \
        https://github.com/Slamtec/rplidar_sdk.git \
        /root/rplidar_sdk \
    && make -C /root/rplidar_sdk -j"$(nproc)"

RUN git clone --branch "${NVBLOX_VERSION}" \
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

RUN rosdep init 2>/dev/null || true \
    && rosdep update

COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENV TENSORRT_ROOT=/opt/tensorrt-10.4 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/opt/tensorrt-10.4/targets/aarch64-linux-gnu/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONPATH=/workspace/Parallax/python:/opt/nanoowl:/opt/torch2trt \
    ROS_DISTRO=humble \
    ROS_DOMAIN_ID=31

RUN ldconfig

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

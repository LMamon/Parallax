FROM dustynv/nanoowl:r36.4.0 AS nanoowl

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive
ARG RPLIDAR_ROS_REF=ros2

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

SHELL ["/bin/zsh", "-c"]

RUN chsh -s /usr/bin/zsh root

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu \
        $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
        > /etc/apt/sources.list.d/ros2.list

# Isaac ROS 3.2 uses the release-3 apt channel on Jammy.
# Install ROS and the accelerated packages as binaries rather than rebuilding
RUN wget -qO - https://isaac.download.nvidia.com/isaac-ros/repos.key | apt-key add - \
    && echo "deb https://isaac.download.nvidia.com/isaac-ros/release-3 $(lsb_release -cs) release-3.0" \
       > /etc/apt/sources.list.d/isaac-ros.list \
    && apt-get update \
    && apt-get install -y \
        python3-ament-package \
        python3-colcon-common-extensions \
        python3-rosdep \
        ros-humble-ament-cmake \
        ros-humble-camera-info-manager \
        ros-humble-cv-bridge \
        ros-humble-foxglove-bridge \
        ros-humble-image-transport \
        ros-humble-isaac-ros-image-proc \
        ros-humble-isaac-ros-nitros \
        ros-humble-isaac-ros-managed-nitros \
        ros-humble-isaac-ros-nitros-image-type \
        ros-humble-isaac-ros-nitros-topic-tools \
        ros-humble-isaac-ros-nvblox \
        ros-humble-isaac-ros-stereo-image-proc \
        ros-humble-isaac-ros-visual-slam \
        ros-humble-nav-msgs \
        ros-humble-ompl \
        ros-humble-ros-base \
        ros-humble-rosidl-typesupport-fastrtps-c \
        ros-humble-rosidl-typesupport-fastrtps-cpp \
        ros-humble-sensor-msgs \
        ros-humble-std-srvs \
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

# Official SLAMTEC ROS 2 driver. master is the ROS 1/catkin package.
RUN mkdir -p /opt/rplidar_ws/src \
    && git clone --depth 1 \
        --branch "${RPLIDAR_ROS_REF}" \
        https://github.com/Slamtec/rplidar_ros.git \
        /opt/rplidar_ws/src/rplidar_ros \
    && source /opt/ros/humble/setup.zsh \
    && cd /opt/rplidar_ws \
    && colcon build --merge-install \
        --packages-select rplidar_ros \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf /opt/rplidar_ws/build /opt/rplidar_ws/log

RUN rosdep init 2>/dev/null || true \
    && rosdep update

COPY docker/ros_entrypoint.zsh /ros_entrypoint.zsh
RUN chmod +x /ros_entrypoint.zsh

ENV TENSORRT_ROOT=/opt/tensorrt-10.4 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/opt/tensorrt-10.4/targets/aarch64-linux-gnu/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONPATH=/workspace/Parallax/python:/opt/nanoowl:/opt/torch2trt \
    ROS_DISTRO=humble \
    ROS_DOMAIN_ID=2 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp

RUN ldconfig

RUN printf '%s\n' \
    'source /opt/ros/humble/setup.zsh' \
    'source /opt/rplidar_ws/install/setup.zsh' \
    'source /workspace/Parallax/ros2_ws/install/setup.zsh' \
    >> /root/.zshrc

ENTRYPOINT ["/ros_entrypoint.zsh"]
CMD ["zsh"]

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

# ros2
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe


RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb


#install ros
RUN apt-get update && \
    apt-get install -y \
        ros-humble-ros-base \
        python3-rosdep \
        python3-colcon-common-extensions \
        python3-vcstool \
        ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

RUN echo "source /opt/ros/humble/setup.zsh" >> /root/.zshrc

# ros2 packages
RUN apt-get update && \
    apt-get install -y \
        ros-humble-cv-bridge \
        ros-humble-foxglove-bridge \
        ros-humble-image-transport \
        ros-humble-launch \
        ros-humble-launch-ros \
        ros-humble-rclcpp \
        ros-humble-rviz2 \
        ros-humble-sensor-msgs \
        ros-humble-std-msgs \
        ros-humble-tf2 \
        ros-humble-tf2-ros \
        ros-humble-tf2-tools \
        ros-humble-xacro \
    && rm -rf /var/lib/apt/lists/*
# pytorch
# tbd
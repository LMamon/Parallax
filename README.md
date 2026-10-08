Perception and spatial planning system for autonomous platforms, developed on NVIDIA Jetson hardware.

It integrates stereo vision, object detection, segmentation, tracking, visual localization, and persistent 3D mapping to support spatial reasoning and collision-aware path planning.

## About

Parallax combines real-time sensor processing and fusion, semantic perception, and geometric mapping within ROS 2 architecture.

The system processes stereo camera and LiDAR data, maintains a localized representation of the environment, and generates 3D navigation paths using observed geometry.

capabilities:

- Stereo image processing, disparity estimation, and metric depth
- Open-vocabulary object detection and image segmentation
- Visual object tracking and 3D spatial association
- Visual SLAM, localization, and relocalization
- 3D reconstruction using TSDF and ESDF representations
- Collision-aware 3D path planning using OMPL
- Independent 2D LiDAR acquisition
- Visualization and interaction through Foxglove

## Hardware

Developed and tested with:

- Jetson Orin Nano 8 GB
- Arducam AR0234 global-shutter stereo camera
- RPLIDAR C1

## Software

- ROS 2 Humble
- NVIDIA JetPack 6.2
- CUDA, TensorRT, and VPI
- NVIDIA Isaac ROS, cuVSLAM, and nvblox
- OMPL
- Docker
- Foxglove

## Installation

### Requirements

- NVIDIA Jetson running a compatible JetPack installation
- Docker with NVIDIA container runtime
- ROS 2 dependencies provided by the project container
- Compatible stereo camera and calibration files

Clone the repository:

```bash
git clone https://github.com/LMamon/Parallax.git
cd Parallax
```

## Usage
Start the containerized runtime:

```bash
./scripts/run.zsh
```

The startup script builds the ROS 2 workspace and launches the configured perception, mapping, localization, planning, and visualization nodes.


## Under Construction

- Autopilot integration for multicopter + rover
- Autonomous tasking and mission management
- Goal management and replanning
- Multi-platform deployment and validation
# Parallax

An unmanned system needs more than a camera feed. It needs to turn
sensor data into useful spatial information: what is visible, where it
is, whether it is moving, and how those observations relate to the
vehicle as it moves through the environment.

Parallax is my implementation of that perception and localization layer
on a Jetson Orin Nano. It uses calibrated stereo cameras and a 2D LiDAR
to produce depth and geometric observations, adds open-vocabulary
detection and optional segmentation, can maintain a selected visual
target, and uses cuVSLAM to place the sensor and observed objects in a
local world frame.

The runtime is built around explicit product dependencies rather than
one fixed processing pipeline. Camera, stereo, neural perception,
tracking, spatial association, localization, and visualization can run
at different rates, and optional work only runs when something actually
requires it. This keeps slower inference and visualization work from
defining the rate of unrelated sensor processing.

## Current capabilities

-   stereo capture, ISP, rectification, disparity, confidence, and
    metric depth;
-   RPLIDAR C1 acquisition with calibrated sensor extrinsics;
-   NanoOWL open-vocabulary object detection;
-   EfficientViT-SAM segmentation when a mask is requested;
-   VPI DCF tracking for a selected target;
-   2D semantic observations associated with stereo/LiDAR measurements
    to produce 3D object positions;
-   NVIDIA cuVSLAM pose estimation, trajectory history, visual
    observations, and landmarks;
-   localized semantic observations in a common world frame;
-   command and subscription-driven execution through Foxglove;
-   bounded product history, source provenance, execution policies, and
    producer-level runtime metrics.

## Next step

Add pathfinding and planning to the existing perception and localization system.

This work remains on the navigation side of the autonomy stack. Vehicle control and actuation are outside the current scope.

## Hardware and software

The current system runs on an NVIDIA Jetson Orin Nano 8 GB with an
Arducam AR0234 global-shutter stereo pair and an RPLIDAR C1.

The runtime is primarily C++17 and uses NVIDIA VPI, CUDA, TensorRT, and
cuVSLAM for the hardware-accelerated portions of the system. Python is
used at the NanoOWL boundary. Foxglove provides the remote command,
debugging, and 2D/3D visualization surface.

## Design

The main design constraint is straightforward: expensive or stateful
perception work should not force the entire system into one lockstep
frame loop.

Products and their dependencies are declared ahead of time. At runtime,
active commands and consumers determine which parts of the graph are
needed. Most realtime edges prefer the newest compatible observation,
while consumers that require temporal continuity, such as cuVSLAM, use
bounded ordered history.

This also keeps the sensor and algorithm boundaries useful. Detection
does not require segmentation. A detection does not need to become a
persistent track before it can be placed in 3D. Foxglove can request or
display a product without owning the computation that creates it.

The current architecture and the reasoning behind the dependency-graph
refactor are documented in
[`docs/architecture.md`](docs/architecture.md) and
[`docs/dependency_graph_refactor.md`](docs/dependency_graph_refactor.md).

## Scope

Parallax currently ends at perception and localization. It does not
contain vehicle actuation.

The output is the part those systems need first: calibrated sensor data,
metric depth, semantic observations, target state, and a local spatial
reference that higher-level autonomy can use to reason about the
environment.

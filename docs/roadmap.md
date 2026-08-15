# Parallax Roadmap

## Runtime Pipeline

introduce a shared runtime layer:

```text
core/
├── runtime
├── pipeline
├── frame
└── module lifecycle
```

The runtime should own orchestration while processing modules remain responsible for their individual operations.

The goal is to support modules that can be enabled or disabled during execution rather than permanently coupling every subsystem to the capture loop.

## Visualization and Bindings

```text
runtime
   ↓
visualization/
   Foxglove
   WebSocket
   ↓
bindings/
   nanobind
```

Python bindings provide access to selected native pipeline components without moving the low-level camera and GPU pipeline into Python.

## Perception

Planned perception work includes:

```text
perception/
├── open-vocabulary detection
├── tracking
├── 3D tracking
└── segmentation
```

RGB imagery supplies appearance-based perception while stereo provides metric spatial information.

## Spatial Pipeline

Future localization work may add VIO and VSLAM when temporal/world-frame reconstruction becomes necessary.

## Multimodal Direction

Longer-term Parallax development can incorporate:

```text
stereo vision
     +
2D LiDAR
     +
camera motion / localization
     +
tracked observations
     ↓
shared spatial representation
```
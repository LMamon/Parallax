# Stereo Pipeline Architecture

## Purpose

This document records the architectural decisions behind the Parallax stereo pipeline established during the `2.0` cycle.

The objective is to maintain a predictable GPU-resident path from raw camera acquisition to disparity and confidence while keeping the individual processing stages independently reusable.

## Camera Acquisition

The stereo camera is accessed directly through V4L2 rather than ROS 2.

The camera produces a combined RAW10 Bayer frame containing the left and right sensor images. Camera configuration and Arducam-specific controls are isolated from downstream image processing.

The camera layer owns acquisition and release of the underlying capture buffers. Downstream components operate on explicit frame representations rather than accessing the camera directly.

## ISP Boundary

The ISP converts the Bayer input into two stereo representations:

Both are produced during the same GPU processing stage.

RGB is retained for consumers such as visualization, object detection, tracking, and other appearance-based tasks.

Grayscale is produced directly for stereo processing. Producing it at the ISP boundary avoids converting RGB back into grayscale later and provides stereo algorithms with their required representation without an additional processing stage.

## GPU Memory

Intermediate image data remains in CUDA memory.

The pipeline avoids unnecessary device-to-host transfers between processing stages. Host downloads are considered observation or debugging operations rather than part of the normal image-processing path.

This allows:

```text
CUDA ISP output > CUDA-backed VPI image > VPI operation
```

without an intermediate image copy.

## VPI Interoperability

NVIDIA VPI is used for GPU rectification and stereo disparity estimation.

CUDA allocations are exposed to VPI through image wrappers. The wrapper owns the VPI image handle but does not own the underlying CUDA allocation.

This distinction is important during shutdown:

```text
VPI wrapper > release wrapper
CUDA buffer > release allocation separately
```

The architecture therefore keeps memory ownership with the Parallax CUDA/frame abstractions while VPI operates on those buffers.

## Rectification

Stereo calibration is performed offline.

Runtime rectification consumes generated calibration artifacts rather than performing calibration itself.

Artifacts include:

- left and right rectification maps
- `R1` and `R2`
- `P1` and `P2`
- `Q`
- calibrated baseline and virtual camera parameters

`StereoCalibration` owns loading and validating these artifacts.

`StereoRectifier` uses the rectification maps to remap left and right images through VPI.

Both RGB and grayscale representations can therefore share the same calibrated stereo geometry.

## Stereo Matching

`StereoMatcher` consumes rectified grayscale images.

The current implementation uses the VPI CUDA Stereo Disparity Estimator with:

```text
Input:          VPI_IMAGE_FORMAT_Y8_ER
Backend:        CUDA
Disparity:      S16
Confidence:     U16
maxDisparity:   128
downscale:      1
```

The matcher produces GPU-resident disparity and confidence buffers.

Depth conversion is intentionally downstream of stereo matching. Disparity represents the direct output of correspondence estimation; depth can subsequently be derived using the calibrated stereo geometry.

## VPI Stream

Rectification and stereo matching use a shared VPI stream.

This provides ordered GPU submission:

```text
rectification > stereo matching > downstream consumer
```

without requiring a host synchronization between every VPI operation.

Components using the shared stream do not own its lifecycle. Stream creation, synchronization, and destruction are handled separately by the application/runtime layer.

Synchronization is required when host code needs to observe the result of asynchronous GPU work.

## Component Ownership

The established boundary is approximately:

```text
Camera
  owns capture lifecycle

ISP
  owns ISP output buffers

StereoCalibration
  owns calibration data

StereoRectifier
  owns rectification resources
  references ISP buffers

StereoMatcher
  owns disparity/confidence buffers
  references rectified buffers

VPI Stream
  owns shared execution stream
```

These ownership boundaries are intended to remain useful when orchestration moves out of `main.cpp`.

## Architectural Boundary

The `2.0` implementation establishes the processing components but still orchestrates them directly from the application entrypoint.

The next refactor changes orchestration rather than the established stereo algorithms:

```text
main.cpp

    ↓

runtime / pipeline
    ├── camera
    ├── ISP
    ├── rectification
    ├── stereo
    └── future modules
```

The stereo implementation documented here therefore serves as the baseline beneath the runtime architecture.
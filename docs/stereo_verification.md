# Stereo Pipeline Verification

Verification performed while establishing the Parallax `2.0` stereo pipeline.

It also records failure modes discovered during integration so that constraints do not have to be rediscovered during later development.

## Verification Environment

Baseline platform:

```text
Platform:       NVIDIA Jetson Orin Nano
OS:             Ubuntu 22.04 / JetPack 6.2.1
CUDA:           12.6
VPI:            3.2
OpenCV:         4.8
Compiler:       GCC 11
Camera:         Arducam AR0234 stereo
Capture:        V4L2
```

Primary stereo capture resolution:

```text
Combined:       3840 × 1200
Per camera:     1920 × 1200
Format:         RAW10 Bayer
Bayer pattern:  GRBG
```

## Camera Verification

Verified:

- V4L2 camera initialization
- RAW10 frame acquisition
- expected frame dimensions
- left/right stereo acquisition
- exposure and analogue gain control
- capture-buffer release
- invalid/warmup frame handling

Camera acquisition is considered operational when valid frames can be repeatedly captured and passed into the ISP without violating the V4L2 buffer lifecycle.

## ISP Verification

Verified GPU processing of the raw Bayer frame into:

```text
RGB Left
RGB Right
Gray Left
Gray Right
```

The grayscale representation is generated during ISP processing rather than reconstructed from downstream RGB images.

Both representations remain CUDA-resident for subsequent processing.

## Calibration Verification

Runtime calibration loading was verified for the generated stereo calibration artifacts.

Expected artifacts include:

```text
R1.npy
R2.npy
P1.npy
P2.npy
Q.npy

left_map_x
left_map_y
right_map_x
right_map_y

metadata
```

The baseline rectification configuration uses:

```text
Image size:  1920 × 1200
Virtual fx:  1100
Virtual fy:  1100
Virtual cx:  960
Virtual cy:  600
```

Calibration and calibration-data generation occur offline. Runtime code only loads and consumes the resulting artifacts.

## Rectification Verification

A known stereo capture was processed through the generated rectification maps.

Verified:

- calibration maps load correctly
- CUDA buffers can be wrapped by VPI
- VPI remap accepts the configured image representation
- left and right images are successfully rectified
- resulting stereo images exhibit expected epipolar alignment
- rectified grayscale output is valid for stereo matching

Rectified image pairs were temporarily downloaded and saved during development for visual inspection.

Those host downloads were verification scaffolding and are not required by the runtime pipeline.

## Stereo Matcher Verification

Stereo matching was verified using:

```text
Backend:        VPI_BACKEND_CUDA
Input:          Y8_ER
Output:         S16 disparity
Confidence:     U16
maxDisparity:   128
downscale:      1
```

Verified:

- rectified grayscale buffers are accepted directly
- disparity estimation submits successfully
- disparity output is produced
- confidence output is produced
- valid disparities occur within the configured 0–128 pixel search range

The VPI S16 disparity representation uses fixed-point disparity values. Host-side validation converted the output to floating-point pixel disparity before inspection.

## Integration Findings

Several integration failures established useful constraints on the pipeline.

### Image Format Compatibility

VPI operations require compatible image formats rather than arbitrary CUDA-backed RGB representations.

Stereo matching ultimately uses rectified `Y8_ER` grayscale images.

This contributed to the decision to expose grayscale as a first-class ISP output rather than perform ad-hoc conversion inside `StereoMatcher`.

### CUDA/VPI Wrapping

VPI image wrappers reference existing CUDA allocations.

The wrapper and underlying allocation therefore have separate lifecycles. Destroying a VPI wrapper must not incorrectly assume ownership of the CUDA memory.

### Asynchronous Execution

VPI GPU work is asynchronous.

Attempting to inspect GPU outputs from the host before the queued work completed produced invalid or misleading results during integration.

The established model is:

```text
submit rectification
        ↓
submit stereo matching
        ↓
shared stream preserves ordering
        ↓
synchronize before host observation
```

GPU stages themselves do not require host synchronization merely to pass data to the next ordered operation on the same stream.

### Disparity Validation

Raw S16 output should not automatically be interpreted as valid scene disparity.

Validation uses the configured disparity range and rejects values outside the expected interval before computing statistics or visualization.

## Temporary Verification Scaffolding

During development, `main.cpp` temporarily performed:

- GPU-to-host disparity downloads
- GPU-to-host confidence downloads
- CUDA synchronization
- disparity conversion
- disparity statistics
- OpenCV color mapping
- disparity PNG output
- confidence PNG output
- single-frame termination

These operations existed to establish that the pipeline worked.

They are not part of the intended runtime architecture and were removed after verification.

## Future Automated Tests

Verification evidence should progressively become repeatable tests where practical.

High-value candidates include:

```text
Unit
├── calibration artifact loading
├── calibration dimensions
├── matrix validation
├── geometry transforms
└── depth conversion

Integration
├── known-pair rectification
├── VPI CUDA wrapper behavior
├── stereo matcher execution
├── disparity range validation
└── confidence output validation

Hardware / System
├── camera initialization
├── live frame acquisition
├── complete GPU pipeline
└── sustained runtime behavior
```

Visual inspection and hardware-dependent behavior do not need to be forced into ordinary unit tests.

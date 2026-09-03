# VPI DCF target validation

Validated on the deployed Jetson Orin Nano environment with VPI 3.2.4.

## Backend

DCF backend = cuda

The installed NVIDIA DCF sample built successfully and processed the complete 421-frame pedestrian sequence with `VPI_BACKEND_CUDA`.

`VPI_BACKEND_PVA` is unavailable on this target and returns:

`VPI_ERROR_INVALID_ARGUMENT: No PVA hardware available`

## Image contract

Parallax `RgbLeft` is CUDA-owned pitch-linear RGB8.

VPI CropScaler 3.2 does not accept RGB8 input. The tracker path requires an explicit device-side RGB8 to RGBA8 conversion before CropScaler.

The existing `parallax::vpi::ImageWrapper` can describe the CUDA allocation without copying it. The tracker path does not require a host image.

## Baseline

sample, 421 frames:

- elapsed: 4.773 s
- average wall time: ~11.34 ms/frame
- throughput: ~88.2 FPS
- observed RAM: ~2827 MB before execution and ~2864–2865 MB during execution
- observed GPU utilization: ~6–17%
- observed system power: ~6.2–6.4 W
- observed GPU temperature: ~54 C

These are whole-sample measurements and include video decode, OpenCV handling, format conversion, synchronization, and other sample work. They are not isolated DCF latency or resource measurements.

The Parallax adapter must measure its RGB8 to RGBA8 conversion and tracker work separately.
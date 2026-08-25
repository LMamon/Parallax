# VPI Buffer Contracts

This document records the accelerator memory ownership and image-layout
contracts at the start of this refactor.

It describes the current implementation before any buffer ownership is
changed.

The purpose is to make each boundary explicit so that later VPI migration
removes unnecessary conversions without breaking CUDA, TensorRT, OFA, or
CPU consumers.

---

## Ownership Rule

Parallax does not use a single allocator for every accelerator product.

Ownership follows the dominant consumer of the product:

- CUDA owns buffers primarily consumed by custom CUDA kernels or future
  TensorRT inference.
- VPI owns intermediates primarily consumed by VPI/VIC/OFA operations.
- CPU ownership is reserved for metadata, control state, calibration data,
  and explicit host-visible outputs.
- VPI wrappers around CUDA allocations are views only. Wrapping a
  `CudaBuffer` does not transfer ownership of the underlying allocation.

The goal is not to convert every image to VPI.

The goal is to make the VPI-heavy portion of the graph naturally
VPI-resident while preserving CUDA-friendly boundaries where they are
useful.

---

## Current Pipeline

in the beginning the relevant image path is:

```text
V4L2 host frame
    |
    v
CUDA pitch-linear Bayer
    |
    | custom CUDA ISP
    v
CUDA pitch-linear RGB + gray
    |
    | VPI wrappers
    v
CUDA pitch-linear rectified RGB + gray
    |
    | gray wrapped by VPI
    | VIC format/layout conversion
    v
VPI Y8_ER_BL left/right
    |
    | OFA stereo
    v
VPI S16_BL disparity
    |
    | VIC format/layout conversion
    v
CUDA pitch-linear S16 disparity
    |
    | custom CUDA kernel
    v
CUDA pitch-linear depth
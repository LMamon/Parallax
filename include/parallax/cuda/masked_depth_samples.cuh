#pragma once

#include <parallax/cuda/cuda_buffer.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace parallax::cuda {

    struct MaskedDepthPoint {
        float x = 0.0F;
        float y = 0.0F;
        float z = 0.0F;
    };

    /**
     * segmentation mask remains in NanoOWL/SAM’s source image space, 
     * while depth is rectified. SegmentationMask explicitly preserves 
     * that image space and remains CUDA-resident.  
     * 
     * The calibration’s rectified→source maps let the GPU ask:
     * for this rectified depth pixel,
     *  which source-image mask pixel corresponds to it?
     * 
     * without uploading the CPU inverse mapper or downloading the mask.
     */

    bool sampleMaskedDepth(const std::uint8_t* mask,
                           std::size_t mask_pitch,
                           std::uint32_t mask_width,
                           std::uint32_t mask_height,
                           const CudaBuffer& depth,
                           const CudaBuffer& rectified_to_rgb_x,
                           const CudaBuffer& rectified_to_rgb_y,
                           float fx,
                           float fy,
                           float cx,
                           float cy,
                           std::uint32_t sample_stride,
                           std::uint32_t max_samples,
                           CudaBuffer& samples,
                           CudaBuffer& sample_count,
                           cudaStream_t stream);

}
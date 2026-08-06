#pragma once

#include <parallax/isp/frame_types.hpp>
#include <cuda_runtime.h>

namespace parallax::isp {

    /**
     * Convert a combined side-by-side BA10 Bayer image into separate,
     * full-resolution RGB8 left and right images.
     * This function only enqueues GPU work. It does not synchronize the stream.
     */
    bool demosaicAndSplit(const GpuBayerFrame& input, StereoRgbFrame& output, cudaStream_t stream);

} // namespace parallax::isp
#pragma once

#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/frame_types.hpp>

#include <parallax/isp/frame_types.hpp>

#include <cuda_runtime.h>

namespace parallax::isp {

    class ISP {
        public:
            ISP();
            ~ISP();

            ISP(const ISP&) = delete;
            ISP& operator=(const ISP&) = delete;

            ISP(ISP&&) = delete;
            ISP& operator=(ISP&&) = delete;

            bool initialize(const parallax::camera::CameraConfig& config); //maybe change interface to params instead of config

            bool process(const parallax::camera::RawFrame& input);

            bool synchronize();
            void shutdown();

            bool downloadRaw(std::uint16_t* host_data, std::size_t host_pitch) const;

            const StereoRgbFrame& rgb() const noexcept { return rgb_output_; }
            const StereoGrayFrame& gray() const noexcept { return gray_output_; }

        private:
            bool upload(const parallax::camera::RawFrame& input);

            // Current ISP work runs on an ISP-owned CUDA stream.
            // keep stream ownership for now; do synchronization later
            cudaStream_t stream_{};

            // CUDA-owned, pitch-linear storage allocated by CudaBuffer via
            // cudaMallocPitch. Raw Bayer data is uploaded from V4L2 and
            // consumed directly by the custom CUDA demosaic/split kernel.
            GpuBayerFrame gpu_input_;

            // CUDA-owned, pitch-linear RGB output.
            //
            // Keep RGB CUDA-friendly because it is an appearance product and
            // future TensorRT/detector consumers should not require a VPI
            // ownership transition. VPI may wrap this storage when needed,
            // but the allocation remains owned by ISP.
            StereoRgbFrame rgb_output_{};

            // CUDA-owned, pitch-linear grayscale output produced by the ISP.
            //
            // Rectification currently wraps these allocations as VPI images
            // without copying or taking ownership. This boundary is a
            // migration candidate because downstream grayscale processing is
            // predominantly VPI/OFA based.
            StereoGrayFrame gray_output_{};

            bool initialized_ = false;
        };
}
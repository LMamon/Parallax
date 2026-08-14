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

            cudaStream_t stream_{};

            GpuBayerFrame gpu_input_;
            StereoRgbFrame rgb_output_{};
            StereoGrayFrame gray_output_{};

            bool initialized_ = false;
        };
}
#pragma once

#include <parallax/stereo/calibration.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/vpi/image_wrapper.hpp>

#include <vpi/Stream.h>
#include <vpi/WarpMap.h>
#include <vpi/algo/Remap.h>

namespace parallax::stereo {
    
    class StereoRectifier {
        public:
            StereoRectifier() = default;
            ~StereoRectifier();

            StereoRectifier(const StereoRectifier&) = delete;
            StereoRectifier& operator=(const StereoRectifier&) = delete;

            bool initialize(const StereoCalibration& calibration,
                            const parallax::isp::StereoRgbFrame& rgb_input,
                            const parallax::isp::StereoGrayFrame& gray_input,
                            VPIStream stream);
            
            bool process(VPIStream stream);
            const parallax::isp::RectifiedStereoFrame& output() const { return output_; } //
            const parallax::isp::RectifiedStereoFrame& rgb() const noexcept { return rgb_output_; }
            const parallax::isp::RectifiedStereoGrayFrame& gray() const noexcept { return gray_output_; }

            void shutdown();
            bool synchronize();
            [[nodiscard]] bool initialized() const noexcept { return initialized_; }
        
        private:
            // Current rectified outputs are CUDA-owned, pitch-linear
            // CudaBuffers. The VPI ImageWrapper members below are non-owning
            // views over these allocations.
            //
            // Phase 7 will reconsider this boundary independently for RGB and
            // grayscale: RGB remains CUDA-friendly for appearance consumers,
            // while rectified grayscale is a candidate for VPI ownership.
            parallax::isp::RectifiedStereoFrame rgb_output_{};
            parallax::isp::RectifiedStereoGrayFrame gray_output_{};

            // Legacy members retained during the ownership audit. Do not
            // remove as part of the documentation-only commit.
            parallax::isp::RectifiedStereoFrame output_{};
            parallax::vpi::ImageWrapper left_input_;
            parallax::vpi::ImageWrapper right_input_;
            parallax::vpi::ImageWrapper left_output_;
            parallax::vpi::ImageWrapper right_output_;

            // Non-owning VPI wrappers over CUDA-owned RGB storage.
            // Inputs are owned by ISP; outputs are owned by StereoRectifier.
            parallax::vpi::ImageWrapper rgb_left_input_;
            parallax::vpi::ImageWrapper rgb_right_input_;
            parallax::vpi::ImageWrapper rgb_left_output_;
            parallax::vpi::ImageWrapper rgb_right_output_;

            // Non-owning VPI wrappers over CUDA-owned pitch-linear Y8 storage.
            // Inputs are owned by ISP; outputs are currently owned by
            // StereoRectifier. Rectified gray ownership is a Phase 7
            // migration target.
            parallax::vpi::ImageWrapper gray_left_input_;
            parallax::vpi::ImageWrapper gray_right_input_;
            parallax::vpi::ImageWrapper gray_left_output_;
            parallax::vpi::ImageWrapper gray_right_output_;

            // CPU-resident VPI warp-map data allocated/freed through the VPI
            // warp-map API. Used to construct the persistent remap payloads.
            VPIWarpMap left_warp_{};
            VPIWarpMap right_warp_{};

            // VPI-owned remap payloads. They operate on the shared VPI stream
            // but do not own that stream.
            VPIPayload left_remap_ = nullptr;
            VPIPayload right_remap_ = nullptr;

            // Borrowed shared execution stream; lifecycle is owned externally.
            VPIStream stream_ = nullptr;
            
            bool left_warp_allocated_ = false;
            bool right_warp_allocated_ = false;
            bool initialized_ = false;
    };
}
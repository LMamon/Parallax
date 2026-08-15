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
            
            bool process();
            const parallax::isp::RectifiedStereoFrame& output() const { return output_; } //
            const parallax::isp::RectifiedStereoFrame& rgb() const noexcept { return rgb_output_; }
            const parallax::isp::RectifiedStereoGrayFrame& gray() const noexcept { return gray_output_; }

            void shutdown();
            bool synchronize();
            [[nodiscard]] bool initialized() const noexcept { return initialized_; }
        
        private:
            parallax::isp::RectifiedStereoFrame rgb_output_{};
            parallax::isp::RectifiedStereoGrayFrame gray_output_{};
            parallax::isp::RectifiedStereoFrame output_{}; //
            // GPU resources
            parallax::vpi::ImageWrapper left_input_; //
            parallax::vpi::ImageWrapper right_input_; //
            parallax::vpi::ImageWrapper left_output_; //
            parallax::vpi::ImageWrapper right_output_; //
            
            // RGB wrappers
            parallax::vpi::ImageWrapper rgb_left_input_;
            parallax::vpi::ImageWrapper rgb_right_input_;
            parallax::vpi::ImageWrapper rgb_left_output_;
            parallax::vpi::ImageWrapper rgb_right_output_;

            parallax::vpi::ImageWrapper gray_left_input_;
            parallax::vpi::ImageWrapper gray_right_input_;
            parallax::vpi::ImageWrapper gray_left_output_;
            parallax::vpi::ImageWrapper gray_right_output_;

            // VPI/CV-CUDA resources
            VPIWarpMap left_warp_{};
            VPIWarpMap right_warp_{};

            // VPI remap resources
            VPIPayload left_remap_ = nullptr;
            VPIPayload right_remap_ = nullptr;
            
            VPIStream stream_ = nullptr;
            
            bool left_warp_allocated_ = false;
            bool right_warp_allocated_ = false;
            bool initialized_ = false;
    };
}
#pragma once

#include <parallax/isp/frame_types.hpp>
#include <parallax/vpi/image_wrapper.hpp>

#include <vpi/Stream.h>
#include <vpi/Types.h>

#include <vpi/algo/StereoDisparity.h>

namespace parallax::stereo {

    class StereoMatcher {
        public:
            StereoMatcher() = default;
            ~StereoMatcher();

            StereoMatcher(const StereoMatcher&) = delete;
            StereoMatcher& operator=(const StereoMatcher&) = delete;

            bool initialize(const parallax::isp::RectifiedStereoGrayFrame& input, VPIStream stream);
            bool process();

            const parallax::isp::StereoMatchFrame& output() const { return output_; }
            void shutdown();

            [[nodiscard]] bool initialized() const noexcept { return initialized_; }
        
        private:
            parallax::isp::StereoMatchFrame output_{};

            // Rectified grayscale input wrappers.
            parallax::vpi::ImageWrapper left_input_;
            parallax::vpi::ImageWrapper right_input_;

            // Stereo outputs.
            parallax::vpi::ImageWrapper disparity_image_;
            parallax::vpi::ImageWrapper confidence_image_;

            VPIPayload stereo_ = nullptr;
            VPIStereoDisparityEstimatorParams submit_params_{};

            VPIStream stream_ = nullptr;
            bool initialized_ = false;
    };
}
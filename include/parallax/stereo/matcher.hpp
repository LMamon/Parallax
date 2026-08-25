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
            // Public stereo-match product currently exposes CUDA-owned,
            // pitch-linear disparity storage so the downstream CUDA depth
            // kernel can consume it directly.
            parallax::isp::StereoMatchFrame output_{};

            // Non-owning VPI wrappers over the rectifier's CUDA-owned,
            // pitch-linear Y8 grayscale inputs.
            parallax::vpi::ImageWrapper left_input_;
            parallax::vpi::ImageWrapper right_input_;

            // Non-owning wrappers over CUDA-owned pitch-linear stereo outputs.
            // disparity_image_ is the current destination of the
            // block-linear -> pitch-linear conversion after OFA.
            parallax::vpi::ImageWrapper disparity_image_;
            parallax::vpi::ImageWrapper confidence_image_;

            // VPI-owned OFA stereo estimator and its reusable submit params.
            VPIPayload stereo_ = nullptr;
            VPIStereoDisparityEstimatorParams submit_params_{};

            // Borrowed shared execution stream; lifecycle is owned externally.
            VPIStream stream_ = nullptr;

            // VPI-owned block-linear images required by the VIC/OFA path.
            //
            // Current path:
            // pitch-linear Y8
            //   -> VIC conversion
            //   -> Y8_ER_BL
            //   -> OFA stereo
            //   -> S16_BL
            //   -> VIC conversion
            //   -> pitch-linear CUDA disparity
            //
            // remove conversions only when downstream ownership/layout
            // contracts make them unnecessary.
            VPIImage left_block_linear_ = nullptr;
            VPIImage right_block_linear_ = nullptr;
            VPIImage disparity_block_linear_ = nullptr;
            bool initialized_ = false;
    };
}
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
            // Published disparity remains CUDA-owned, pitch-linear S16 because
            // the downstream depth implementation is a custom CUDA kernel.
            parallax::isp::StereoMatchFrame output_{};
            // Non-owning VPI views over rectifier-owned CUDA pitch-linear
            // VPI_IMAGE_FORMAT_Y8_ER images. Wrapping does not copy or transfer
            // ownership of the underlying CUDA allocations.
            parallax::vpi::ImageWrapper left_input_;
            parallax::vpi::ImageWrapper right_input_;
            // Non-owning VPI view over the published CUDA S16 disparity buffer.
            // VIC writes the OFA block-linear result into this pitch-linear
            // representation before publication.
            parallax::vpi::ImageWrapper disparity_image_;
            VPIPayload stereo_ = nullptr;
            VPIStereoDisparityEstimatorParams submit_params_{};
            // Borrowed shared VPI stream. StereoMatcher does not own or destroy
            // the stream.
            VPIStream stream_ = nullptr;

            // 8-bit extended-range grayscale, block-linear.
            //
            // The OFA backend requires block-linear stereo input in VPI 3.2.
            // VIC converts the rectified pitch-linear Y8_ER images into these
            // VPI-owned buffers without CPU staging.
            VPIImage left_block_linear_ = nullptr;
            VPIImage right_block_linear_ = nullptr;

            // Signed Q10.5 disparity, block-linear.
            //
            // OFA-only stereo requires VPI_IMAGE_FORMAT_S16_BL output.
            // The result is converted by VIC to pitch-linear S16 only because
            // the downstream custom CUDA depth kernel requires CUDA-accessible
            // pitch-linear storage.
            VPIImage disparity_block_linear_ = nullptr;
            bool initialized_ = false;
    };
}
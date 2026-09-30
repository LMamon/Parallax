#pragma once

#include <parallax/core/completion.hpp>
#include <parallax/core/fixed_payload_pool.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/vpi/image_wrapper.hpp>

#include <memory>
#include <vpi/Stream.h>
#include <vpi/algo/StereoDisparity.h>

namespace parallax::core {
class ExecutionContext;
}

namespace parallax::stereo {

class StereoMatcher {
public:
    StereoMatcher() = default;
    ~StereoMatcher();

    StereoMatcher(const StereoMatcher&) = delete;
    StereoMatcher& operator=(const StereoMatcher&) = delete;

    static constexpr std::size_t OutputSlotCount = 5;

    struct OutputSlot {
        parallax::isp::StereoMatchFrame output{};
        parallax::vpi::ImageWrapper left_input;
        parallax::vpi::ImageWrapper right_input;
        parallax::vpi::ImageWrapper disparity_image;
        parallax::vpi::ImageWrapper confidence_image;
        parallax::core::CompletionHandle completion{};
    };

    bool initialize(const parallax::isp::RectifiedStereoGrayFrame& input,
                    VPIStream stream);

    [[nodiscard]] std::shared_ptr<OutputSlot> acquireOutput(
        parallax::core::ExecutionContext& context);

    bool process(const parallax::isp::RectifiedStereoGrayFrame& input,
                 OutputSlot& output,
                 VPIStream stream);
    
    [[nodiscard]] std::shared_ptr<OutputSlot> acquireOutputSynchronous();
    
    void shutdown();
    [[nodiscard]] bool initialized() const noexcept { return initialized_; }

private:
    parallax::core::FixedPayloadPool<OutputSlot, OutputSlotCount> output_pool_;
    VPIPayload stereo_ = nullptr;
    VPIStereoDisparityEstimatorParams submit_params_{};
    VPIStream stream_ = nullptr;
    bool initialized_ = false;
};

}  // namespace parallax::stereo

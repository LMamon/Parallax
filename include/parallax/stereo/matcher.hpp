#pragma once

#include <parallax/isp/frame_types.hpp>
#include <parallax/vpi/image_wrapper.hpp>
#include <parallax/core/fixed_payload_pool.hpp>
#include <parallax/core/completion.hpp>

#include <memory>
#include <vpi/Stream.h>
#include <vpi/Types.h>

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

            bool initialize(const parallax::isp::RectifiedStereoGrayFrame& input, VPIStream stream);
            static constexpr std::size_t OutputSlotCount = 3;

            struct OutputSlot {
                parallax::isp::StereoMatchFrame output{};

                /**
                 * Per-submission VPI views over the exact rectified input generation consumed
                 * by this slot.
                 *
                 * VPI may retain these wrapper containers after C++ consumers have released
                 * the corresponding product. The slot therefore carries its previous VPI
                 * completion handle; acquireOutput() waits for that completion before these
                 * wrappers may be rebound.
                 *
                 * The wrappers do not own the underlying rectified CUDA allocations.
                 * StereoProducer keeps that input generation alive through the published
                 * product lifetime dependency.
                 */
                parallax::vpi::ImageWrapper left_input;
                parallax::vpi::ImageWrapper right_input;

                /**
                 * Completion fence for the most recent VPI submission that used this slot.
                 *
                 * C++ shared ownership prevents reuse while a published product still leases
                 * the slot. This handle separately prevents rebinding the slot's VPI wrapper
                 * containers while the accelerator may still have them locked.
                 */
                parallax::core::CompletionHandle completion{};

                VPIImage left_block_linear = nullptr;
                VPIImage right_block_linear = nullptr;
                VPIImage disparity_block_linear = nullptr;

                parallax::vpi::ImageWrapper disparity_image;

                ~OutputSlot() {
                    if (left_block_linear != nullptr) {
                        vpiImageDestroy(left_block_linear);
                        left_block_linear = nullptr;
                    }

                    if (right_block_linear != nullptr) {
                        vpiImageDestroy(right_block_linear);
                        right_block_linear = nullptr;
                    }

                    if (disparity_block_linear != nullptr) {
                        vpiImageDestroy(disparity_block_linear);
                        disparity_block_linear = nullptr;
                    }
                }

                OutputSlot() = default;

                OutputSlot(const OutputSlot&) = delete;
                OutputSlot& operator=(const OutputSlot&) = delete;
            };

            [[nodiscard]] std::shared_ptr<OutputSlot> acquireOutput(parallax::core::ExecutionContext& context);
            /**
            * Asynchronous graph acquisition path.
            *
            * Waits for the selected slot's previous accelerator submission before its
            * VPI wrappers may be rebound.
            */
            [[nodiscard]] std::shared_ptr<OutputSlot> acquireOutput() {
                return output_pool_.acquire();
            }

            bool process(const parallax::isp::RectifiedStereoGrayFrame& input, OutputSlot& output, VPIStream stream);

            const parallax::isp::StereoMatchFrame& output() const noexcept {
                return output_pool_.prototype()->output;
            }
            void shutdown();

            [[nodiscard]] bool initialized() const noexcept { return initialized_; }
        
        private:
            parallax::core::FixedPayloadPool<OutputSlot, OutputSlotCount> output_pool_;
            VPIPayload stereo_ = nullptr;
            VPIStereoDisparityEstimatorParams submit_params_{};
            // Borrowed shared VPI stream. StereoMatcher does not own or destroy
            // the stream.
            VPIStream stream_ = nullptr;
            bool initialized_ = false;
    };
}
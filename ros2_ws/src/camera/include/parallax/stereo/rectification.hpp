#pragma once

#include <parallax/stereo/calibration.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/vpi/image_wrapper.hpp>
#include <parallax/core/fixed_payload_pool.hpp>

#include <memory>
#include <vpi/Stream.h>
#include <vpi/WarpMap.h>
#include <vpi/algo/Remap.h>
#include <array>
#include <cstddef>


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
            // cuVSLAM retains up to 8 ordered rectified generations.
            // One additional slot is required so the next generation can be
            // produced before ProductStore evicts the oldest retained generation.
            static constexpr std::size_t OutputSlotCount = 9;

            struct OutputSlot {
                parallax::isp::RectifiedStereoFrame rgb{};
                parallax::isp::RectifiedStereoGrayFrame gray{};

                // VPI container identity belongs to the storage generation.
                // These wrappers remain permanently paired with this slot's CUDA buffers.
                parallax::vpi::ImageWrapper rgb_left_wrapper;
                parallax::vpi::ImageWrapper rgb_right_wrapper;
                parallax::vpi::ImageWrapper gray_left_wrapper;
                parallax::vpi::ImageWrapper gray_right_wrapper;
            };

            [[nodiscard]] std::shared_ptr<OutputSlot> acquireOutput() {
                return output_pool_.acquire();
            }

            bool process(const parallax::isp::StereoRgbFrame& rgb_input,
                         const parallax::isp::StereoGrayFrame& gray_input, 
                         OutputSlot& output, 
                         VPIStream stream);


            const parallax::isp::RectifiedStereoFrame& rgb() const noexcept {
                if (latest_output_ != nullptr) return latest_output_->rgb;

                return output_pool_.prototype()->rgb;
            }

            const parallax::isp::RectifiedStereoGrayFrame& gray() const noexcept {
                return output_pool_.prototype()->gray;
            }

            void shutdown();
            [[nodiscard]] bool initialized() const noexcept { return initialized_; }
        
        private:
            struct InputWrapperSet {
                parallax::vpi::ImageWrapper rgb_left;
                parallax::vpi::ImageWrapper rgb_right;
                parallax::vpi::ImageWrapper gray_left;
                parallax::vpi::ImageWrapper gray_right;

                [[nodiscard]] bool valid() const noexcept {
                    return rgb_left.valid() && rgb_right.valid() && gray_left.valid() && gray_right.valid();
                }

                void release() noexcept {
                    rgb_left.release();
                    rgb_right.release();
                    gray_left.release();
                    gray_right.release();
                }
            };

            bool ensureInputWrappers(const parallax::isp::StereoRgbFrame& rgb_input,
                                     const parallax::isp::StereoGrayFrame& gray_input);

            std::array<InputWrapperSet, OutputSlotCount> input_wrappers_{};
            /**
             * Non-owning pointer to the most recently submitted output slot.
             *
             * Lifetime remains governed by the fixed pool and published Product handles.
             * This exists only to preserve the established visualization accessor while
             * output ownership becomes generation-specific.
             */
            const OutputSlot* latest_output_ = nullptr;

            /*
            * Rectification storage and VPI container identity are generation-specific.
            * Input wrappers are indexed by the ISP storage slot; output wrappers are owned
            * directly by each fixed rectification output slot.
            */
            parallax::core::FixedPayloadPool<OutputSlot, OutputSlotCount> output_pool_;

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
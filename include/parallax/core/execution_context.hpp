#pragma once

#include <parallax/core/product_store.hpp>
#include <parallax/vpi/stream.hpp>

#include <chrono>

namespace parallax::core {

    /**
     * Runtime-scoped owner for resources shared across graph execution.
     *
     * ExecutionContext is intentionally separate from Producer and from the
     * algorithm implementations. Producers describe graph work; algorithm classes
     * continue to own algorithm-specific payloads/resources; this context owns
     * infrastructure shared by multiple producer families.
     * 
     * Execution lanes are named by independent work domain rather than by class.
     * Multiple producers may therefore share a lane when their work is ordered by
     * dependency, while independent branches remain able to overlap.
     
     * this object grows incrementally:
     *   - completed product storage and common timing utilities
     *   - named VPI execution lanes
     *   - CUDA/TensorRT execution resources
     *   - bounded CPU execution resources
     *
     * Context lifetime is intended to match Runtime lifetime.
     */
    class ExecutionContext {
        public:
            using Clock = std::chrono::steady_clock;
            using TimePoint = Clock::time_point;

            ExecutionContext() = default;
            ~ExecutionContext();

            ExecutionContext(const ExecutionContext&) = delete;
            ExecutionContext& operator=(const ExecutionContext&) = delete;

            ExecutionContext(ExecutionContext&&) = delete;
            ExecutionContext& operator=(ExecutionContext&&) = delete;

            /**
             * Create the shared execution resources owned by this context.
             *
             * This currently initializes only VPI lanes. Later Phase 6 work adds
             * neural/CUDA and CPU execution resources behind the same lifecycle.
             */
            bool initialize();

            /**
             * Release execution resources in deterministic order.
             *
             * Individual algorithm payloads remain owned by their algorithm
             * classes; only shared execution infrastructure is released here.
             */
            void shutdown() noexcept;

            [[nodiscard]] ProductStore& products() noexcept {
                return product_store_;
            }

            [[nodiscard]] const ProductStore& products() const noexcept {
                return product_store_;
            }

            [[nodiscard]] static TimePoint now() noexcept {
                return Clock::now();
            }

            /**
             * Independent image work receives separate left/right lanes so future
             * preprocessing can overlap when no graph dependency orders the eyes.
             */
            [[nodiscard]] parallax::vpi::Stream& imageLeftLane() noexcept {
                return image_left_lane_;
            }

            [[nodiscard]] parallax::vpi::Stream& imageRightLane() noexcept {
                return image_right_lane_;
            }

            /**
             * Shared image preprocessing that is not intrinsically left- or
             * right-eye work uses this lane rather than acquiring a class-specific
             * stream.
             */
            [[nodiscard]] parallax::vpi::Stream& preprocessLane() noexcept {
                return preprocess_lane_;
            }

            /**
             * Stereo receives a dedicated lane because VIC format conversion and
             * OFA disparity form an ordered accelerator chain.
             */

            [[nodiscard]] parallax::vpi::Stream& stereoLane() noexcept {
                return stereo_lane_;
            }

            /**
             * Conventional pose / geometry work may execute independently from
             * stereo once its required image product is complete.
             */

            [[nodiscard]] parallax::vpi::Stream& conventionalPoseLane() noexcept {
                return conventional_pose_lane_;
            }

            [[nodiscard]] const parallax::vpi::Stream& imageLeftLane() const noexcept {
                return image_left_lane_;
            }

            [[nodiscard]] const parallax::vpi::Stream& imageRightLane() const noexcept {
                return image_right_lane_;
            }

            [[nodiscard]] const parallax::vpi::Stream& preprocessLane() const noexcept {
                return preprocess_lane_;
            }

            [[nodiscard]] const parallax::vpi::Stream& stereoLane() const noexcept {
                return stereo_lane_;
            }

            [[nodiscard]] const parallax::vpi::Stream& conventionalPoseLane() const noexcept {
                return conventional_pose_lane_;
            }

            [[nodiscard]] bool initialized() const noexcept {
                return initialized_;
            }

        private:
            ProductStore product_store_;

            // These lanes represent execution independence, not producer identity.
            // Event dependencies between them are added later rather than forcing
            // unrelated work through one globally serialized VPI stream.
            parallax::vpi::Stream image_left_lane_;
            parallax::vpi::Stream image_right_lane_;
            parallax::vpi::Stream preprocess_lane_;
            parallax::vpi::Stream stereo_lane_;
            parallax::vpi::Stream conventional_pose_lane_;

            bool initialized_ = false;      
    };
}
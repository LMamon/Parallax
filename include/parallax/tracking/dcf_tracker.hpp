#pragma once

#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/vpi/image_wrapper.hpp>

#include <opencv2/core/types.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Types.h>
#include <vpi/algo/DCFTracker.h>


namespace parallax::tracking {

    struct DcfTrackerResult {
        cv::Rect2f box{};
        float response{0.0F};
        bool tracked{false};
    };

    class DcfTracker {
        public:
            DcfTracker() = default;
            ~DcfTracker();

            DcfTracker(const DcfTracker&) = delete;
            DcfTracker& operator=(const DcfTracker&) = delete;

            bool initialize(const parallax::cuda::CudaBuffer& image, const cv::Rect2f& box);
            [[nodiscard]] DcfTrackerResult update(const parallax::cuda::CudaBuffer& image);

            void reset() noexcept;
            [[nodiscard]] bool initialized() const noexcept { return initialized_; }

        private:
            bool ensureResources(const parallax::cuda::CudaBuffer& image);
            bool bindInput(const parallax::cuda::CudaBuffer& image);
            bool convertInput();
            bool crop(VPIArray objects);
            bool sync();
            bool readResult(DcfTrackerResult& result);
            void logVpiError(const char* message, VPIStatus status) const;

            parallax::vpi::ImageWrapper input_wrapper_{};

            VPIStream stream_{nullptr};
            VPIPayload crop_scaler_{nullptr};
            VPIPayload dcf_{nullptr};

            VPIImage rgba_frame_{nullptr};
            VPIImage target_patch_{nullptr};

            VPIArray in_targets_{nullptr};
            VPIArray out_targets_{nullptr};
            VPIArray max_responses_{nullptr};

            VPIDCFTrackerCreationParams creation_params_{};
            VPIDCFTrackerParams params_{};

            std::uint32_t width_{0};
            std::uint32_t height_{0};
            int32_t patch_size_{0};

            bool initialized_{false};
    };
}
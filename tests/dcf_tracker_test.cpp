#include <gtest/gtest.h>

#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/tracking/dcf_tracker.hpp>

#include <cuda_runtime.h>

namespace {

    TEST(DcfTrackerTest, InitializeUpdateResetOnCudaResidentFrame) {
        constexpr std::uint32_t width = 640;
        constexpr std::uint32_t height = 480;

        parallax::cuda::CudaBuffer frame;

        ASSERT_TRUE(frame.allocate(width, height, 3, sizeof(std::uint8_t)));
        ASSERT_EQ(cudaMemset2D(frame.data(), frame.pitch(), 127, width * 3, height), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        parallax::tracking::DcfTracker tracker;

        const cv::Rect2f initial_box{220.0F,
                                     140.0F,
                                     120.0F,
                                     160.0F};

        ASSERT_TRUE(tracker.initialize(frame, initial_box));
        EXPECT_TRUE(tracker.initialized());

        const auto result = tracker.update(frame);

        // A uniform synthetic frame is not a tracking-quality test. This only
        // exercises the CUDA/VPI path and persistent tracker resources.
        EXPECT_GE(result.box.width, 0.0F);
        EXPECT_GE(result.box.height, 0.0F);

        tracker.reset();

        EXPECT_FALSE(tracker.initialized());
    }
}
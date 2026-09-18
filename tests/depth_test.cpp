#include <gtest/gtest.h>

#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/cuda/depth.cuh>
#include <parallax/stereo/depth_policy.hpp>

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>

namespace {

    TEST(DepthTest, RejectsGeometryOutsideUsefulStereoRange) {
        constexpr std::uint32_t Width = 6;
        constexpr std::uint32_t Height = 1;

        // fx=100px and baseline=0.1m gives Z=10/disparity.
        const std::array<std::int16_t, Width> disparity{
            20,  // 0.5m -> too close
            10,  // 1.0m -> valid
            2,   // 5.0m -> valid
            1,   // 10m  -> too far
            0,   // invalid disparity
            -1   // invalid disparity
        };

        parallax::cuda::CudaBuffer disparity_device;
        parallax::cuda::CudaBuffer depth_device;

        ASSERT_TRUE(disparity_device.allocate(Width, Height, 1, sizeof(std::int16_t)));
        ASSERT_TRUE(depth_device.allocate(Width, Height, 1, sizeof(float)));

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        ASSERT_TRUE(disparity_device.uploadAsync(disparity.data(), Width * sizeof(std::int16_t), stream));

        ASSERT_TRUE(parallax::cuda::disparityToDepth(
            disparity_device, depth_device, 100.0F, 0.1F, 1.0F,
            parallax::stereo::MinUsefulDepthM,
            parallax::stereo::MaxUsefulDepthM,
            stream));

        std::array<float, Width> depth{};
        ASSERT_TRUE(depth_device.downloadAsync(depth.data(), Width * sizeof(float), stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        EXPECT_TRUE(std::isnan(depth[0]));
        EXPECT_FLOAT_EQ(depth[1], 1.0F);
        EXPECT_FLOAT_EQ(depth[2], 5.0F);
        EXPECT_TRUE(std::isnan(depth[3]));
        EXPECT_TRUE(std::isnan(depth[4]));
        EXPECT_TRUE(std::isnan(depth[5]));

        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST(DepthTest, RejectsInvalidRangeConfiguration) {
        parallax::cuda::CudaBuffer disparity_device;
        parallax::cuda::CudaBuffer depth_device;
        ASSERT_TRUE(disparity_device.allocate(1, 1, 1, sizeof(std::int16_t)));
        ASSERT_TRUE(depth_device.allocate(1, 1, 1, sizeof(float)));

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        EXPECT_FALSE(parallax::cuda::disparityToDepth(
            disparity_device, depth_device, 100.0F, 0.1F, 1.0F,
            parallax::stereo::MaxUsefulDepthM,
            parallax::stereo::MinUsefulDepthM,
            stream));

        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST(DepthTest, RejectsMismatchedBuffers) {
        parallax::cuda::CudaBuffer disparity_device;
        parallax::cuda::CudaBuffer depth_device;
        ASSERT_TRUE(disparity_device.allocate(2, 1, 1, sizeof(std::int16_t)));
        ASSERT_TRUE(depth_device.allocate(1, 1, 1, sizeof(float)));

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        EXPECT_FALSE(parallax::cuda::disparityToDepth(
            disparity_device, depth_device, 100.0F, 0.1F, 1.0F,
            parallax::stereo::MinUsefulDepthM,
            parallax::stereo::MaxUsefulDepthM,
            stream));

        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

}

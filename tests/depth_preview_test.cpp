#include <gtest/gtest.h>

#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/cuda/depth_preview.cuh>

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>

namespace {

    TEST(DepthPreviewTest, UsesLocalConsensusAndRejectsOutlier) {
        constexpr std::uint32_t SW = 4, SH = 4, Stride = 2, PW = 2, PH = 2;
        const std::array<float, SW * SH> source{
            1.00F, 1.02F, 2.00F, 2.02F,
            0.99F, 8.00F, 1.98F, 2.01F,
            3.00F, 3.02F, 4.00F, 4.02F,
            2.98F, 3.01F, 4.01F, 9.00F
        };

        parallax::cuda::CudaBuffer src, dst;
        ASSERT_TRUE(src.allocate(SW, SH, 1, sizeof(float)));
        ASSERT_TRUE(dst.allocate(PW, PH, 1, sizeof(float)));

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        ASSERT_TRUE(src.uploadAsync(source.data(), SW * sizeof(float), stream));
        ASSERT_TRUE(parallax::cuda::downsampleDepthRobust(src, dst, Stride, stream));

        std::array<float, PW * PH> result{};
        ASSERT_TRUE(dst.downloadAsync(result.data(), PW * sizeof(float), stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        EXPECT_NEAR(result[0], (1.00F + 1.02F + 0.99F) / 3.0F, 0.001F);
        EXPECT_NEAR(result[1], (2.00F + 2.02F + 1.98F + 2.01F) / 4.0F, 0.001F);
        EXPECT_NEAR(result[2], (3.00F + 3.02F + 2.98F + 3.01F) / 4.0F, 0.001F);
        EXPECT_NEAR(result[3], (4.00F + 4.02F + 4.01F) / 3.0F, 0.001F);

        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST(DepthPreviewTest, IgnoresInvalidSamplesAndFillsSmallPreviewHole) {
        const std::array<float, 4> source{NAN, 1.50F, 1.52F, NAN};

        parallax::cuda::CudaBuffer src, dst;
        ASSERT_TRUE(src.allocate(2, 2, 1, sizeof(float)));
        ASSERT_TRUE(dst.allocate(1, 1, 1, sizeof(float)));

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        ASSERT_TRUE(src.uploadAsync(source.data(), 2 * sizeof(float), stream));
        ASSERT_TRUE(parallax::cuda::downsampleDepthRobust(src, dst, 2, stream));

        float result = NAN;
        ASSERT_TRUE(dst.downloadAsync(&result, sizeof(float), stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        EXPECT_NEAR(result, 1.51F, 0.001F);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST(DepthPreviewTest, LeavesUnsupportedBlockUnknown) {
        const std::array<float, 4> source{NAN, NAN, 2.00F, NAN};

        parallax::cuda::CudaBuffer src, dst;
        ASSERT_TRUE(src.allocate(2, 2, 1, sizeof(float)));
        ASSERT_TRUE(dst.allocate(1, 1, 1, sizeof(float)));

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        ASSERT_TRUE(src.uploadAsync(source.data(), 2 * sizeof(float), stream));
        ASSERT_TRUE(parallax::cuda::downsampleDepthRobust(src, dst, 2, stream));

        float result = 0.0F;
        ASSERT_TRUE(dst.downloadAsync(&result, sizeof(float), stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        EXPECT_TRUE(std::isnan(result));
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST(DepthPreviewTest, SupportsNonDivisibleSourceDimensions) {
        parallax::cuda::CudaBuffer src, dst;
        ASSERT_TRUE(src.allocate(5, 3, 1, sizeof(float)));
        ASSERT_TRUE(dst.allocate(3, 2, 1, sizeof(float)));

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        EXPECT_TRUE(parallax::cuda::downsampleDepthRobust(src, dst, 2, stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST(DepthPreviewTest, RejectsInvalidDestinationShape) {
        parallax::cuda::CudaBuffer src, dst;
        ASSERT_TRUE(src.allocate(8, 4, 1, sizeof(float)));
        ASSERT_TRUE(dst.allocate(8, 4, 1, sizeof(float)));

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        EXPECT_FALSE(parallax::cuda::downsampleDepthRobust(src, dst, 2, stream));
        EXPECT_FALSE(parallax::cuda::downsampleDepthRobust(src, dst, 0, stream));
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

}

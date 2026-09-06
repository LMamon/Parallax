#include <gtest/gtest.h>

#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/cuda/masked_depth_samples.cuh>

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

    constexpr std::uint32_t Width = 8;
    constexpr std::uint32_t Height = 8;
    constexpr std::uint32_t MaxSamples = 64;

    TEST(MaskedDepthSamplesTest, SamplesOnlyMaskSupportedValidDepth) {
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        std::vector<float> depth(Width * Height, 2.0F);
        std::vector<float> map_x(Width * Height);
        std::vector<float> map_y(Width * Height);
        std::vector<std::uint8_t> mask(Width * Height, 0);

        /*
        * Identity rectified->RGB mapping. Only the central 4x4 source-image
        * region belongs to the segmentation mask.
        */
        for (std::uint32_t y = 0; y < Height; ++y) {
            for (std::uint32_t x = 0; x < Width; ++x) {
                const std::size_t index = static_cast<std::size_t>(y) * Width + x;

                map_x[index] = static_cast<float>(x);
                map_y[index] = static_cast<float>(y);

                if (x >= 2 && x <= 5 && y >= 2 && y <= 5) {
                    mask[index] = 255;
                }
            }
        }

        parallax::cuda::CudaBuffer depth_device;
        parallax::cuda::CudaBuffer map_x_device;
        parallax::cuda::CudaBuffer map_y_device;
        parallax::cuda::CudaBuffer samples_device;
        parallax::cuda::CudaBuffer count_device;

        ASSERT_TRUE(depth_device.allocate(Width, Height, 1, sizeof(float)));

        ASSERT_TRUE(map_x_device.allocate(Width, Height, 1, sizeof(float)));

        ASSERT_TRUE(map_y_device.allocate(Width, Height, 1, sizeof(float)));

        ASSERT_TRUE(samples_device.allocate(MaxSamples, 1, 1, sizeof(parallax::cuda::MaskedDepthPoint)));
        ASSERT_TRUE(count_device.allocate(1, 1, 1, sizeof(std::uint32_t)));

        ASSERT_TRUE(depth_device.uploadAsync(depth.data(), Width * sizeof(float), stream));
        ASSERT_TRUE(map_x_device.uploadAsync(map_x.data(), Width * sizeof(float), stream));

        ASSERT_TRUE(map_y_device.uploadAsync(map_y.data(), Width * sizeof(float), stream));

        std::uint8_t* mask_device = nullptr;

        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&mask_device), mask.size()), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(mask_device, mask.data(), mask.size(), cudaMemcpyHostToDevice, stream), cudaSuccess);

        ASSERT_TRUE(parallax::cuda::sampleMaskedDepth(mask_device,
                                                      Width,
                                                      Width,
                                                      Height,
                                                      depth_device,
                                                      map_x_device,
                                                      map_y_device,
                                                      100.0F,
                                                      100.0F,
                                                      4.0F,
                                                      4.0F,
                                                      1,
                                                      MaxSamples,
                                                      samples_device,
                                                      count_device,
                                                      stream));

        std::array<parallax::cuda::MaskedDepthPoint, MaxSamples> samples{};

        std::uint32_t count = 0;

        ASSERT_TRUE(samples_device.downloadAsync(samples.data(), sizeof(samples), stream));

        ASSERT_TRUE(count_device.downloadAsync(&count, sizeof(count), stream));

        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(count, 16U);

        /*
        * Every accepted sample came from valid 2 m stereo depth underneath
        * the segmentation mask.
        */
        for (std::size_t i = 0; i < count; ++i) {
            EXPECT_FLOAT_EQ(samples[i].z, 2.0F);

            EXPECT_GE(samples[i].x, -0.04F);
            EXPECT_LE(samples[i].x, 0.02F);

            EXPECT_GE(samples[i].y, -0.04F);
            EXPECT_LE(samples[i].y, 0.02F);
        }

        ASSERT_EQ(cudaFree(mask_device), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
}
#include <parallax/core/runtime_metrics.hpp>
#include <parallax/cuda/cuda_buffer.cuh>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace parallax::core {
    namespace {

    TEST(RuntimeMetricsTest, CudaBufferCountsAllocationAndFree) {
        reset_runtime_metrics();

        {
            parallax::cuda::CudaBuffer buffer;

            ASSERT_TRUE(buffer.allocate(16, 8, 1, sizeof(std::uint8_t)));

            const auto& metrics = runtime_metrics();

            EXPECT_EQ(metrics.cuda_allocations.load(), 1);
            EXPECT_GT(metrics.cuda_allocated_bytes.load(), 0);
            EXPECT_EQ(metrics.cuda_frees.load(), 0);
        }

        EXPECT_EQ(runtime_metrics().cuda_frees.load(), 1);
    }

    TEST(RuntimeMetricsTest, CudaBufferCountsHostDeviceTransfers) {
        reset_runtime_metrics();

        parallax::cuda::CudaBuffer buffer;
        ASSERT_TRUE(buffer.allocate(8, 4, 1, sizeof(std::uint8_t)));

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        std::array<std::uint8_t, 32> host{};

        ASSERT_TRUE(buffer.uploadAsync(host.data(), 8, stream));

        ASSERT_TRUE(buffer.downloadAsync(host.data(), 8, stream));

        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

        const auto& metrics = runtime_metrics();

        EXPECT_EQ(metrics.host_to_device_transfers.load(), 1);
        EXPECT_EQ(metrics.host_to_device_bytes.load(), 32);

        EXPECT_EQ(metrics.device_to_host_transfers.load(), 1);
        EXPECT_EQ(metrics.device_to_host_bytes.load(), 32);
    }

    TEST(RuntimeMetricsTest, CudaBufferCountsDeviceCopies) {
        reset_runtime_metrics();

        parallax::cuda::CudaBuffer source;
        parallax::cuda::CudaBuffer destination;

        ASSERT_TRUE(source.allocate(8, 4, 1, sizeof(std::uint8_t)));
        ASSERT_TRUE(destination.allocate(8, 4, 1, sizeof(std::uint8_t)));

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        ASSERT_TRUE(destination.copyFromAsync(source, stream));

        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

        const auto& metrics = runtime_metrics();

        EXPECT_EQ(metrics.device_to_device_transfers.load(), 1);
        EXPECT_EQ(metrics.device_to_device_bytes.load(), 32);
    }

    }
}
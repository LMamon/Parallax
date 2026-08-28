#include <parallax/core/runtime_metrics.hpp>
#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/core/execution_context.hpp>

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

        TEST(RuntimeMetricsTest, HostWaitIsCounted) {
            reset_runtime_metrics();

            ExecutionContext context;
            ASSERT_TRUE(context.initialize());

            cudaStream_t stream = nullptr;
            ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

            auto completion = context.recordCudaCompletion(stream);
            ASSERT_TRUE(completion.valid());

            ASSERT_TRUE(context.waitForHost(completion));

            EXPECT_EQ(runtime_metrics().host_waits.load(), 1);

            ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }

        TEST(RuntimeMetricsTest, ContextDrainIsCounted) {
            reset_runtime_metrics();

            ExecutionContext context;
            ASSERT_TRUE(context.initialize());

            ASSERT_TRUE(context.drain());

            EXPECT_EQ(runtime_metrics().context_drains.load(), 1);
        }

        TEST(RuntimeMetricsTest, AcceleratorWaitIsCounted) {
            reset_runtime_metrics();

            ExecutionContext context;
            ASSERT_TRUE(context.initialize());

            cudaStream_t producer_stream = nullptr;
            ASSERT_EQ(cudaStreamCreate(&producer_stream), cudaSuccess);

            auto completion = context.recordCudaCompletion(producer_stream);
            ASSERT_TRUE(completion.valid());

            ASSERT_TRUE(context.waitFor(completion, context.preprocessLane()));

            EXPECT_EQ(runtime_metrics().accelerator_waits.load(), 1);
            EXPECT_EQ(runtime_metrics().host_waits.load(), 0);

            ASSERT_TRUE(context.drain());
            ASSERT_EQ(cudaStreamDestroy(producer_stream), cudaSuccess);
        }
        
    }
}
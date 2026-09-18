#include <gtest/gtest.h>
#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/cuda/depth_preview.cuh>
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
namespace {
TEST(DepthPreviewTest, SamplesOneDepthPerStrideOnDevice) {
    constexpr std::uint32_t SW=8, SH=4, Stride=2, PW=4, PH=2;
    const std::array<float,SW*SH> source{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                                        16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    parallax::cuda::CudaBuffer src,dst;
    ASSERT_TRUE(src.allocate(SW,SH,1,sizeof(float)));
    ASSERT_TRUE(dst.allocate(PW,PH,1,sizeof(float)));
    cudaStream_t stream=nullptr; ASSERT_EQ(cudaStreamCreate(&stream),cudaSuccess);
    ASSERT_TRUE(src.uploadAsync(source.data(),SW*sizeof(float),stream));
    ASSERT_TRUE(parallax::cuda::downsampleDepthNearest(src,dst,Stride,stream));
    std::array<float,PW*PH> preview{};
    ASSERT_TRUE(dst.downloadAsync(preview.data(),PW*sizeof(float),stream));
    ASSERT_EQ(cudaStreamSynchronize(stream),cudaSuccess);
    const std::array<float,PW*PH> expected{0,2,4,6,16,18,20,22};
    EXPECT_EQ(preview,expected); ASSERT_EQ(cudaStreamDestroy(stream),cudaSuccess);
}
TEST(DepthPreviewTest, SupportsNonDivisibleSourceDimensions) {
    parallax::cuda::CudaBuffer src,dst; ASSERT_TRUE(src.allocate(5,3,1,sizeof(float)));
    ASSERT_TRUE(dst.allocate(3,2,1,sizeof(float))); cudaStream_t stream=nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream),cudaSuccess);
    EXPECT_TRUE(parallax::cuda::downsampleDepthNearest(src,dst,2,stream));
    ASSERT_EQ(cudaStreamSynchronize(stream),cudaSuccess); ASSERT_EQ(cudaStreamDestroy(stream),cudaSuccess);
}
TEST(DepthPreviewTest, RejectsInvalidDestinationShape) {
    parallax::cuda::CudaBuffer src,dst; ASSERT_TRUE(src.allocate(8,4,1,sizeof(float)));
    ASSERT_TRUE(dst.allocate(8,4,1,sizeof(float))); cudaStream_t stream=nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream),cudaSuccess);
    EXPECT_FALSE(parallax::cuda::downsampleDepthNearest(src,dst,2,stream));
    EXPECT_FALSE(parallax::cuda::downsampleDepthNearest(src,dst,0,stream));
    ASSERT_EQ(cudaStreamDestroy(stream),cudaSuccess);
}
}

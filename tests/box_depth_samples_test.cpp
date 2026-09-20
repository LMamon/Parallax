#include <gtest/gtest.h>
#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/cuda/masked_depth_samples.cuh>
#include <cuda_runtime.h>
#include <array>
#include <vector>
namespace {
TEST(BoxDepthSamplesTest,UsesMappedSourceBox){
 constexpr std::uint32_t W=8,H=8,N=64; cudaStream_t stream=nullptr; ASSERT_EQ(cudaStreamCreate(&stream),cudaSuccess);
 std::vector<float> z(W*H,2.0F),mx(W*H),my(W*H); for(std::uint32_t y=0;y<H;++y)for(std::uint32_t x=0;x<W;++x){auto i=y*W+x;mx[i]=x;my[i]=y;}
 parallax::cuda::CudaBuffer d,xm,ym,s,c; ASSERT_TRUE(d.allocate(W,H,1,sizeof(float)));ASSERT_TRUE(xm.allocate(W,H,1,sizeof(float)));ASSERT_TRUE(ym.allocate(W,H,1,sizeof(float)));ASSERT_TRUE(s.allocate(N,1,1,sizeof(parallax::cuda::MaskedDepthPoint)));ASSERT_TRUE(c.allocate(1,1,1,sizeof(std::uint32_t)));
 ASSERT_TRUE(d.uploadAsync(z.data(),W*sizeof(float),stream));ASSERT_TRUE(xm.uploadAsync(mx.data(),W*sizeof(float),stream));ASSERT_TRUE(ym.uploadAsync(my.data(),W*sizeof(float),stream));
 ASSERT_TRUE(parallax::cuda::sampleBoxDepth(2,2,3,3,d,xm,ym,100,100,4,4,1,N,s,c,stream)); std::uint32_t n=0; ASSERT_TRUE(c.downloadAsync(&n,sizeof(n),stream)); ASSERT_EQ(cudaStreamSynchronize(stream),cudaSuccess); EXPECT_EQ(n,16U); ASSERT_EQ(cudaStreamDestroy(stream),cudaSuccess);
}}

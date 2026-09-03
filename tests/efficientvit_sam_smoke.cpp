#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/efficientvit_sam.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

    void check_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
            std::exit(1);
        }
    }

}


int main() {
    using parallax::perception::EfficientVitSam;
    using parallax::perception::EfficientVitSamResult;

    constexpr std::uint32_t width = 1000;
    constexpr std::uint32_t height = 577;

    std::vector<std::uint8_t> host(static_cast<std::size_t>(width) *
                                   height *
                                   parallax::isp::StereoRgbFrame::Channels,
                                   0);

    cudaStream_t stream = nullptr;

    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    parallax::isp::StereoRgbFrame frame{};

    if (!frame.left.allocate(width,
                             height,
                             parallax::isp::StereoRgbFrame::Channels,
                             sizeof(std::uint8_t))) {

        std::cerr << "failed to allocate input frame\n";
        cudaStreamDestroy(stream);
        return 1;
    }

    if (!frame.left.uploadAsync(host.data(),
                                static_cast<std::size_t>(width) *
                                parallax::isp::StereoRgbFrame::Channels,
                                stream)) {

        std::cerr << "failed to upload input frame\n";
        cudaStreamDestroy(stream);
        return 1;
    }

    frame.width = width;
    frame.height = height;

    EfficientVitSam sam;

    const std::filesystem::path encoder = "models/efficientvit-sam/engines/l0_encoder_fp16.engine";
    const std::filesystem::path decoder = "models/efficientvit-sam/engines/l0_decoder_fp16.engine";

    if (!sam.initialize(encoder, decoder)) {
        std::cerr << "failed to initialize EfficientViT-SAM\n";
        cudaStreamDestroy(stream);
        return 1;
    }

    EfficientVitSamResult result;

    const cv::Rect2f box{75.0F,
                         95.0F,
                         290.0F,
                         255.0F};

    if (!sam.segment(frame, box, stream, result)) {
        std::cerr << "segmentation submission failed\n";

        sam.shutdown();
        cudaStreamDestroy(stream);

        return 1;
    }

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    if (!result.valid()) {
        std::cerr << "invalid segmentation result\n";

        sam.shutdown();
        cudaStreamDestroy(stream);

        return 1;
    }

    const auto metrics = sam.metrics();

    std::cout << "input pitch: " << frame.left.pitch() << '\n';
    std::cout << "mask: " << result.width << 'x' << result.height << '\n';
    std::cout << "mask pitch: " << result.pitch_bytes << '\n';
    std::cout << "inferences: " << metrics.inference_count << '\n';
    std::cout << "PASS: native EfficientViT-SAM adapter\n";

    
    // Release the mask lease before shutting down the backend pool.
     
    result = {};
    sam.shutdown();
    cudaStreamDestroy(stream);

    return 0;
}
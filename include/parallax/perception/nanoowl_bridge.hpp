#pragma once

#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace parallax::perception {

    struct NanoOwlMetrics {
        std::uint64_t query_revision = 0;
        std::uint64_t query_encoding_count = 0;
        double last_predict_ms = 0.0;

    };

    class NanoOwlBridge {
        public:
            NanoOwlBridge();
            ~NanoOwlBridge();

            NanoOwlBridge(const NanoOwlBridge&) = delete;
            NanoOwlBridge& operator=(const NanoOwlBridge&) = delete;

            bool initialize(const std::filesystem::path& engine_path);
            bool setQuery(const std::string& query, std::uint64_t revision);

            /**
             * Run NanoOWL against one CUDA-resident RGB generation.
             *
             * The source allocation remains owned by the Product generation.
             * PyTorch receives a non-owning strided CUDA tensor view.
             *
             * Only compact detection metadata crosses back to host memory.
             */
            bool predict(const parallax::isp::StereoRgbFrame& frame,
                         cudaStream_t stream,
                         DetectionSet& detections);

            void shutdown() noexcept;

            [[nodiscard]] bool initialized() const noexcept {
                return initialized_;
            }

            [[nodiscard]] NanoOwlMetrics metrics() const noexcept;

        private:
            class Impl;
            std::unique_ptr<Impl> impl_;

            bool initialized_ = false;
    };
}
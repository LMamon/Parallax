#pragma once

#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace parallax::perception {

    class NanoOwlBridge {
        public:
            NanoOwlBridge();
            ~NanoOwlBridge();

            NanoOwlBridge(const NanoOwlBridge&) = delete;
            NanoOwlBridge& operator=(const NanoOwlBridge&) = delete;

            bool initialize(const std::filesystem::path& engine_path);
            bool setQuery(const std::string& query, std::uint64_t revision);

            /**
             * Run NanoOWL against one exact CUDA-resident RGB generation.
             *
             * The input allocation remains owned by the source Product generation.
             * PyTorch receives a non-owning strided view over that allocation.
             *
             * Only compact CPU detection metadata crosses back into Parallax.
             */
            bool predict(const parallax::isp::StereoRgbFrame& frame,
                         cudaStream_t stream,
                         DetectionSet& detections);

            void shutdown() noexcept;

            [[nodiscard]] bool initialized() const noexcept {
                return initialized_;
            }

        private:
            class Impl;
            std::unique_ptr<Impl> impl_;

            bool initialized_ = false;
    };
}
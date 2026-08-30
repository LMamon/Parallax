#pragma once

#include <parallax/isp/frame_types.hpp>

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
             * Exercise the zero-copy CUDA RGB boundary.
             *
             * The source allocation remains owned by the
             * StereoRgbFrame/Product generation.
             *
             * PyTorch receives a non-owning strided view.
             */
            bool predict(const parallax::isp::StereoRgbFrame& frame, cudaStream_t stream);

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
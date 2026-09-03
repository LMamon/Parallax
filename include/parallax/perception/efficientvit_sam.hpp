#pragma once

#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>

namespace parallax::perception {

    struct EfficientVitSamMetrics {
        double last_encoder_ms = 0.0;
        double last_decoder_ms = 0.0;
        std::uint64_t inference_count = 0;
    };

    struct EfficientVitSamGeometry {
        int original_width;
        int original_height;

        int encoder_width;
        int encoder_height;

        int prompt_width;
        int prompt_height;
    };

    struct EfficientVitSamResult {
        std::shared_ptr<const void> storage{};

        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::size_t pitch_bytes = 0;

        float confidence = 0.0F;
        
        [[nodiscard]] bool valid() const noexcept {
            return static_cast<bool>(storage) && width != 0 && height != 0 && pitch_bytes != 0;
        }
    };

    class EfficientVitSam {
        public:
            EfficientVitSam();
            ~EfficientVitSam();

            EfficientVitSam(const EfficientVitSam&) = delete;
            EfficientVitSam& operator=(const EfficientVitSam&) = delete;

            bool initialize(const std::filesystem::path& encoder_engine,
                            const std::filesystem::path& decoder_engine);

            bool segment(const parallax::isp::StereoRgbFrame& frame,
                         const cv::Rect2f& box,
                         cudaStream_t stream,
                         EfficientVitSamResult& result);
                         
            void shutdown() noexcept;

            [[nodiscard]] bool initialized() const noexcept {
                return initialized_;
            }

            [[nodiscard]] EfficientVitSamMetrics metrics() const noexcept;

        private:
            class Impl;
            std::unique_ptr<Impl> impl_;

            bool initialized_ = false;
    };
}
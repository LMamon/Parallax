#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct _GstElement;

namespace parallax::visualization {
    class VideoEncoder {
        public:
            VideoEncoder() = default;
            ~VideoEncoder();

            VideoEncoder(const VideoEncoder&) = delete;
            VideoEncoder& operator=(const VideoEncoder&) = delete;

            bool initialize(std::uint32_t width, std::uint32_t height, std::uint32_t fps);
            bool encode(const std::uint8_t* rgb, std::size_t size, std::vector<std::byte>& encoded);

            void shutdown();

            [[nodiscard]] bool initialized() const noexcept { return initialized_; }
        
        private:
            _GstElement* pipeline_ = nullptr;
            _GstElement* appsrc_ = nullptr;
            _GstElement* appsink_ = nullptr;

            std::uint32_t width_ = 0;
            std::uint32_t height_ = 0;
            std::uint32_t fps_ = 0;

            std::uint32_t frame_index_ = 0;

            bool initialized_ = false;
    };
}

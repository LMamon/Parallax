#pragma once

#include <parallax/camera/frame_types.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <linux/videodev2.h>

namespace parallax::camera {

    struct Buffer {
        void* start = nullptr;
        std::size_t length = 0;
        std::uint32_t index = 0;
    };

    class V4L2Device {
        public:
            explicit V4L2Device(const std::string& device);
            virtual ~V4L2Device();

            V4L2Device(const V4L2Device&) = delete;
            V4L2Device& operator=(const V4L2Device&) = delete;

            bool open();
            void close();

            [[nodiscard]] bool isOpen() const noexcept;

            bool setFormat(std::uint32_t width, std::uint32_t height, std::uint32_t fourcc);

            bool setControl(std::uint32_t id, std::int32_t value);
            bool getControl(std::uint32_t id, std::int32_t& value);

            bool initializeStreaming(std::uint32_t buffer_count = 2);
            void shutdownStreaming();

            bool startStreaming();
            void stopStreaming();

            [[nodiscard]] bool isStreaming() const noexcept;

            bool dequeue(RawFrame& frame, int timeout_ms);
            bool queue(const RawFrame& frame);

            [[nodiscard]] std::uint32_t getPixelFormat();
            [[nodiscard]] std::vector<std::uint32_t> getPixelFormats() const;

            [[nodiscard]] std::vector<Resolution> getFrameSizes(std::uint32_t pixel_format) const;

        protected:
            [[nodiscard]] int fileDescriptor() const noexcept {
                return fd_;
            }

            std::string device_;

        private:
            [[nodiscard]] std::chrono::nanoseconds toTimestamp(const timeval& timestamp) const noexcept;

            int fd_ = -1;

            std::vector<Buffer> buffers_;

            std::uint32_t width_ = 0;
            std::uint32_t height_ = 0;
            std::uint32_t fourcc_ = 0;

            bool streaming_ = false;
    };

}
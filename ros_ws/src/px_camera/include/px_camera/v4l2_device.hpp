#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <chrono>

#include <linux/videodev2.h>

namespace px_camera {
    struct FrameSize {
        std::uint32_t width;
        std::uint32_t height;
    };

    struct RawFrame {
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t fourcc;

        const std::uint16_t* data = nullptr;
        std::size_t bytes = 0;
        std::chrono::nanoseconds timestamp{0};
        uint32_t buffer_index;
    };
    
    struct Buffer {
        void* start = nullptr;
        std::size_t length = 0;

        uint32_t index = 0;
    };
    
    class V4L2Device {
        public:
        explicit V4L2Device(const std::string& device);
        virtual ~V4L2Device();
        
        V4L2Device(const V4L2Device&) = delete;
        V4L2Device& operator=(const V4L2Device&) = delete;
        
        bool open();
        void close();
        
        bool isOpen() const;
        
        bool setFormat(std::uint32_t width, std::uint32_t height, std::uint32_t fourcc);
        bool setControl(std::uint32_t id, std::int32_t value);
        
        bool initializeStreaming(std::uint32_t buffer_count = 4);
        void shutdownStreaming();
        bool startStreaming();
        void stopStreaming();
        bool isStreaming() const;
        
        bool dequeue(RawFrame& frame, int timeout_ms = 3000);
        bool queue(const RawFrame& frame);
        
        std::uint32_t getPixelFormat() const;
        std::vector<std::uint32_t> getPixelFormats() const;
        std::vector<FrameSize> getFrameSizes(std::uint32_t pixel_format) const;
        
    protected:
        std::string device_;
        
    private:
        int fd_ = -1;
        
        std::vector<Buffer> buffers_;
        
        std::uint32_t width_ = 0;
        std::uint32_t height_ = 0;
        std::uint32_t fourcc_ = 0;
        std::size_t buffer_count_ = 0;
        bool streaming_ = false;
        std::chrono::nanoseconds toTimestamp(const timeval& tv) const;
    };
}
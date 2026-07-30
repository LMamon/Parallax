#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <linux/videodev2.h>

namespace px_camera {
    struct FrameSize {
        std::uint32_t width;
        std::uint32_t height;
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

        std::uint32_t getPixelFormat() const;
        std::vector<std::uint32_t> getPixelFormats() const;
        std::vector<FrameSize> getFrameSizes(std::uint32_t pixel_format) const;
        
    protected:
        int fd_ = -1;
        std::string device_;
    };
}
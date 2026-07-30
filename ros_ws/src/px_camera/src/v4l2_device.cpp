#include <px_camera/v4l2_device.hpp>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>

namespace px_camera {
    V4L2Device::V4L2Device(const std::string& device): device_(device) {}
    V4L2Device::~V4L2Device() {
        close();
    }

    bool V4L2Device::open() {
        // O_RDWR is required for most V4L2 ioctl operations
        fd_ = ::open(device_.c_str(), O_RDWR);

        if (fd_ < 0) {
            std::cerr << "Failed to open " << device_ << ": " << std::strerror(errno) << '\n';
            return false;
        }
        return true;
    }

    void V4L2Device::close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool V4L2Device::isOpen() const {
        return fd_ >= 0;
    }

    std::uint32_t V4L2Device::getPixelFormat() const {
        if (!isOpen()) return 0;

        //ask the driver for the camera's current capture format
        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if (::ioctl(fd_, VIDIOC_G_FMT, &format) < 0) {
            std::cerr << "VIDIOC_G_FMT faied: " << std::strerror(errno) << "\n";
            return 0;
        }
        return format.fmt.pix.pixelformat;
    }

    std::vector<std::uint32_t> V4L2Device::getPixelFormats() const {
        std::vector<std::uint32_t> formats;

        if (!isOpen()) return formats;

        //enumerate every pixel format advertised by the driver
        v4l2_fmtdesc description{};
        description.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        for (description.index = 0;; ++description.index) {
            if (::ioctl(fd_, VIDIOC_ENUM_FMT, &description) < 0) {
                // EINVAL means there are no more formats to enumerate
                if (errno != EINVAL) {
                    std::cerr << "VIDIOC_ENUM_FMT failed: " << std::strerror(errno) << '\n';
            }
            break;
        }
        formats.push_back(description.pixelformat);
    }
    return formats;
}

std::vector<FrameSize> V4L2Device::getFrameSizes(std::uint32_t pixel_format) const {
    std::vector<FrameSize> sizes;
    
    if (!isOpen()) return sizes;

    //query the resolutions supported for this specific pixel format.
    v4l2_frmsizeenum frame_size{};
    frame_size.pixel_format = pixel_format;

    for (frame_size.index = 0;; ++ frame_size.index) {
        if (::ioctl(fd_, VIDIOC_ENUM_FRAMESIZES, &frame_size) < 0) {
            if (errno != EINVAL) {
                std::cerr << "VIDIOC_ENUM_FRAMESIZES failed " << std::strerror(errno) << "\n";
            }
            break;
        }
        if (frame_size.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
            sizes.push_back({frame_size.discrete.width, frame_size.discrete.height});
        }
    }
    return sizes;
}
}
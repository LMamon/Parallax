#include <parallax/camera/v4l2_device.hpp>
#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/logger.hpp>

#include <cerrno>
#include <iostream>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace parallax::camera {

    V4L2Device::V4L2Device(const std::string& device) : device_(device) {}

    V4L2Device::~V4L2Device() { close(); }
    parallax::camera::CameraConfig config;
    bool V4L2Device::open() {
        if (isOpen()) {
            logMessage("open: device is already open");
            return true;
        }

        // O_RDWR is required for most V4L2 ioctl operations.
        fd_ = ::open(device_.c_str(), O_RDWR | O_CLOEXEC);

        if (fd_ < 0) {
            logError("open", device_.c_str());
            return false;
        }

        return true;
    }

    void V4L2Device::close() {
        if (!isOpen()) return;

        stopStreaming();
        shutdownStreaming();

        if (::close(fd_) < 0) logError("close", device_.c_str());
        fd_ = -1;
    }

    bool V4L2Device::isOpen() const noexcept {
        return fd_ >= 0;
    }

    std::uint32_t V4L2Device::getPixelFormat() {
        if (!isOpen()) return 0;

        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if (::ioctl(fd_, VIDIOC_G_FMT, &format) < 0) {
            logError("VIDIOC_G_FMT");
            return 0;
        }

        width_ = format.fmt.pix.width;
        height_ = format.fmt.pix.height;
        fourcc_ = format.fmt.pix.pixelformat;

        return fourcc_;
    }

    std::vector<std::uint32_t> V4L2Device::getPixelFormats() const {
        std::vector<std::uint32_t> formats;

        if (!isOpen()) return formats;

        v4l2_fmtdesc description{};
        description.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        for (description.index = 0;; ++description.index) {
            if (::ioctl(fd_, VIDIOC_ENUM_FMT, &description) < 0) {
                if (errno != EINVAL) {
                    logError("VIDIOC_ENUM_FMT");
                }
                break;
            }
            formats.push_back(description.pixelformat);
        }
        return formats;
    }

    std::vector<Resolution> V4L2Device::getFrameSizes(std::uint32_t pixel_format) const {
        std::vector<Resolution> sizes;
        if (!isOpen()) return sizes;

        v4l2_frmsizeenum frame_size{};
        frame_size.pixel_format = pixel_format;

        for (frame_size.index = 0;; ++frame_size.index) {
            if (::ioctl(fd_, VIDIOC_ENUM_FRAMESIZES, &frame_size) < 0) {
                if (errno != EINVAL) {
                    logError("VIDIOC_ENUM_FRAMESIZES");
                }

                break;
            }

            if (frame_size.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                Resolution resolution{};
                resolution.width = frame_size.discrete.width;
                resolution.height = frame_size.discrete.height;

                sizes.push_back(resolution);
            }
        }
        return sizes;
    }

    bool V4L2Device::setFormat(std::uint32_t width, std::uint32_t height, std::uint32_t fourcc) {
        if (!isOpen()) {
            logMessage("setFormat: device is not open");
            return false;
        }

        if (!buffers_.empty() || streaming_) {
            logMessage("setFormat: streaming resources are already initialized");
            return false;
        }

        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = width;
        format.fmt.pix.height = height;
        format.fmt.pix.pixelformat = fourcc;
        format.fmt.pix.field = V4L2_FIELD_NONE;

        if (::ioctl(fd_, VIDIOC_S_FMT, &format) < 0) {
            logError("VIDIOC_S_FMT");
            return false;
        }

        // Store the format actually accepted by the driver.
        width_ = format.fmt.pix.width;
        height_ = format.fmt.pix.height;
        fourcc_ = format.fmt.pix.pixelformat;

        return true;
    }

    bool V4L2Device::setControl(std::uint32_t id, std::int32_t value) {
        if (!isOpen()) {
            logMessage("setControl: device is not open");
            return false;
        }

        v4l2_control control{};
        control.id = id;
        control.value = value;

        if (::ioctl(fd_, VIDIOC_S_CTRL, &control) < 0) {
            logError("VIDIOC_S_CTRL");
            return false;
        }

        return true;
    }

    bool V4L2Device::getControl(std::uint32_t id, std::int32_t& value) {
        v4l2_control control{};
        control.id = id;

        if (::ioctl(fd_, VIDIOC_G_CTRL, &control) < 0)
            return false;

        value = control.value;
        return true;
    }

    bool V4L2Device::initializeStreaming(std::uint32_t buffer_count) {
        if (!isOpen()) {
            logMessage("initializeStreaming: device is not open");
            return false;
        }

        if (streaming_) {
            logMessage("initializeStreaming: stream is already running");
            return false;
        }

        if (!buffers_.empty()) {
            logMessage("initializeStreaming: buffers are already initialized");
            return false;
        }

        if (width_ == 0 || height_ == 0 || fourcc_ == 0) {
            logMessage("initializeStreaming: format has not been configured");
            return false;
        }

        if (buffer_count == 0) {
            logMessage("initializeStreaming: buffer count must be greater than zero");
            return false;
        }

        v4l2_requestbuffers request{};
        request.count = buffer_count;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP;

        if (::ioctl(fd_, VIDIOC_REQBUFS, &request) < 0) {
            logError("VIDIOC_REQBUFS");
            return false;
        }

        if (request.count == 0) {
            logMessage("VIDIOC_REQBUFS: driver allocated zero buffers");
            return false;
        }

        buffers_.reserve(request.count);

        for (std::uint32_t index = 0; index < request.count; ++index) {
            v4l2_buffer buffer{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = index;

            if (::ioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0) {
                logError("VIDIOC_QUERYBUF");
                shutdownStreaming();
                return false;
            }

            void* address = ::mmap(nullptr,
                                    buffer.length,
                                    PROT_READ | PROT_WRITE,
                                    MAP_SHARED,
                                    fd_,
                                    buffer.m.offset);

            if (address == MAP_FAILED) {
                logError("mmap");
                shutdownStreaming();
                return false;
            }

            Buffer mapped_buffer{};
            mapped_buffer.start = address;
            mapped_buffer.length = buffer.length;
            mapped_buffer.index = index;

            buffers_.push_back(mapped_buffer);
        }

        // Every mapped buffer must be queued before STREAMON.
        for (const Buffer& mapped : buffers_) {
            v4l2_buffer buffer{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = mapped.index;

            if (::ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
                logError("VIDIOC_QBUF");
                shutdownStreaming();
                return false;
            }
        }

        return true;
    }

    bool V4L2Device::startStreaming() {
        if (!isOpen()) {
            logMessage("startStreaming: device is not open");
            return false;
        }

        if (streaming_) return true;

        if (buffers_.empty()) {
            logMessage("startStreaming: buffers are not initialized");
            return false;
        }

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if (::ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            logError("VIDIOC_STREAMON");
            return false;
        }

        streaming_ = true;
        return true;
    }

    bool V4L2Device::dequeue(RawFrame& frame) {
        /*
        * Perform a single dequeue attempt.
        *
        * Startup retry policy belongs in StereoCamera::warmup().
        * Runtime retry policy belongs to the caller.
        */
        if (!isOpen() || !streaming_) {
            logMessage("dequeue: stream is not running");
            return false;
        }

        pollfd descriptor{};
        descriptor.fd = fd_;
        descriptor.events = POLLIN | POLLPRI;

        int poll_result;

        do {
            poll_result = ::poll(&descriptor, 1, config.frame_timeout);
        } while (poll_result < 0 && errno == EINTR);

        if (poll_result == 0) {
            logMessage("poll: timeout waiting for frame");
            return false;
        }

        if (poll_result < 0) {
            logError("poll");
            return false;
        }

        if (descriptor.events & (POLLERR | POLLHUP | POLLNVAL)) {
            logMessage("poll: camera device reported a stream error");
            return false;
        }

        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;

        while (true) {
            if (::ioctl(fd_, VIDIOC_DQBUF, &buffer) == 0) break;
            if (errno = EINTR) continue;
            if (errno = EAGAIN) return false;

            logError("VIDIOC_DQBUF");
            return false;
        }

        if (buffer.index >= buffers_.size()) {
            logMessage("VIDIOC_DQBUF: driver returned invalid buffer index");
            return false;
        }

        Buffer& mapped = buffers_[buffer.index];

        if (buffer.bytesused > mapped.length) {
            logMessage("VIDIOC_DQBUF: bytesused exceeds mapped buffer length");
            if (::ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) logError("VIDIOC_QBUF after invalid frame size");
            
            return false;
        }

        if (buffer.flags & V4L2_BUF_FLAG_ERROR) {
            if (::ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
                logError("VIDIOC_QBUF after erroneous frame");
                return false;
            }
            logMessage("VIDIOC_DQBUF: driver returned an error frame");
            return false;
        }
        
        frame.width = width_;
        frame.height = height_;
        frame.fourcc = fourcc_;
        frame.data = mapped.start;
        frame.bytes = buffer.bytesused;
        frame.timestamp = toTimestamp(buffer.timestamp);
        frame.buffer_index = buffer.index;

        return true;   
    }

    bool V4L2Device::queue(const RawFrame& frame) {
        if (!isOpen() || !streaming_) {
            logMessage("queue: stream is not running");
            return false;
        }

        if (frame.buffer_index >= buffers_.size()) {
            logMessage("queue: invalid buffer index");
            return false;
        }

        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = frame.buffer_index;

        if (::ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
            logError("VIDIOC_QBUF");
            return false;
        }

        return true;
    }

    void V4L2Device::stopStreaming() {
        if (!isOpen() || !streaming_) return;

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if (::ioctl(fd_, VIDIOC_STREAMOFF, &type) < 0) {
            logError("VIDIOC_STREAMOFF");
        }

        streaming_ = false;
    }

    void V4L2Device::shutdownStreaming() {
        if (streaming_) stopStreaming();

        for (Buffer& buffer : buffers_) {
            if (buffer.start != nullptr && buffer.start != MAP_FAILED) {
                if (::munmap(buffer.start, buffer.length) < 0) {
                    logError("munmap");
                }

                buffer.start = nullptr;
                buffer.length = 0;
            }
        }

        buffers_.clear();
        if (!isOpen()) return;

        // count = 0 asks the driver to release its MMAP buffers.
        v4l2_requestbuffers request{};
        request.count = 0;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP;

        if (::ioctl(fd_, VIDIOC_REQBUFS, &request) < 0) {
            logError("VIDIOC_REQBUFS release");
        }
    }

    bool V4L2Device::isStreaming() const noexcept {
        return streaming_;
    }

    std::chrono::nanoseconds V4L2Device::toTimestamp(const timeval& timestamp) const noexcept {
        using namespace std::chrono;

        return duration_cast<nanoseconds>(seconds(timestamp.tv_sec) + microseconds(timestamp.tv_usec));
    }
} 
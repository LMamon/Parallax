#include <parallax/vpi/stream.hpp>

#include <vpi/Status.h>

#include <iostream>
#include <utility>

namespace parallax::vpi {

namespace {

    void logVpiError(const char* message, VPIStatus status) {
        char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH]{};
        vpiGetLastStatusMessage(buffer, sizeof(buffer));

        std::cerr
            << message << ": "
            << vpiStatusGetName(status) << " - "
            << buffer << '\n';
    }
}

    Stream::~Stream() { shutdown(); }
    Stream::Stream(Stream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

    Stream& Stream::operator=(Stream&& other) noexcept {
        if (this != &other) {
            shutdown();
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }

    bool Stream::initialize(std::uint64_t backends) {
        if (stream_ != nullptr) { return true; }

        const VPIStatus status = vpiStreamCreate(backends, &stream_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create VPI stream", status);
            stream_ = nullptr;
            return false;
        }
        return true;
    }

    bool Stream::synchronize() {
        if (stream_ == nullptr) { return false; }

        const VPIStatus status = vpiStreamSync(stream_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to synchronize VPI stream", status);
            return false;
        }
        return true;
    }

    void Stream::shutdown() noexcept {
        if (stream_ != nullptr) {
            vpiStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }
}
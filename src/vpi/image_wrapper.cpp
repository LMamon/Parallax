#include <parallax/vpi/image_wrapper.hpp>

#include <iostream>
#include <utility>

namespace parallax::vpi {
    ImageWrapper::~ImageWrapper() { release(); }
    ImageWrapper::ImageWrapper(ImageWrapper&& other) noexcept : image_(std::exchange(other.image_, nullptr)) {}
    
    ImageWrapper& ImageWrapper::operator=(ImageWrapper&& other) noexcept {
        if (this != &other) {
            release();
            image_ = std::exchange(other.image_, nullptr);
        }
        return *this;
    }

    bool ImageWrapper::create(const parallax::cuda::CudaBuffer& buffer, VPIImageFormat format) {
        release();

        if (!buffer.isAllocated()) {
            std::cerr << "Cannot wrap unallocated CUDA buffer\n";
            return false;
        }

        VPIImageData data{};
        data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;

        auto& pitch = data.buffer.pitch;

        pitch.format = format;
        pitch.numPlanes = 1;

        pitch.planes[0].data = const_cast<void*>(buffer.data());
        pitch.planes[0].width = static_cast<int32_t>(buffer.width());
        pitch.planes[0].height = static_cast<int32_t>(buffer.height());
        pitch.planes[0].pitchBytes = static_cast<int32_t>(buffer.pitch());

        VPIImageWrapperParams params {};
        VPIStatus status = vpiInitImageWrapperParams(&params);

        if (status != VPI_SUCCESS) {
            std::cerr << "Failed to initialize VPI image wrapper parameters\n";
            return false;
        }

        if (format == VPI_IMAGE_FORMAT_Y8_ER) params.colorSpec = VPI_COLOR_SPEC_BT601_ER;

        status = vpiImageCreateWrapper(&data, &params, VPI_BACKEND_CUDA | VPI_BACKEND_VIC, &image_);
        if (status != VPI_SUCCESS) {
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH]{};
            vpiGetLastStatusMessage(buffer, sizeof(buffer));

            std::cerr << "Failed to create VPI image wrapper: "
                    << vpiStatusGetName(status)
                    << " - " << buffer << '\n';

            image_ = nullptr;
            return false;
        }
        return true;
    }

    bool ImageWrapper::rebind(const parallax::cuda::CudaBuffer& buffer, VPIImageFormat format) {
        if (image_ == nullptr) return create(buffer, format);

        if (!buffer.isAllocated()) {
            std::cerr << "Cannot rebind VPI wrapper to unallocated CUDA buffer\n";
            return false;
        }

        VPIImageData data{};
        data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;

        auto& pitch = data.buffer.pitch;

        pitch.format = format;
        pitch.numPlanes = 1;

        pitch.planes[0].data = const_cast<void*>(buffer.data());
        pitch.planes[0].width = static_cast<int32_t>(buffer.width());
        pitch.planes[0].height = static_cast<int32_t>(buffer.height());
        pitch.planes[0].pitchBytes = static_cast<int32_t>(buffer.pitch());

        VPIStatus status = vpiImageSetWrapper(image_, &data);

        if (status != VPI_SUCCESS) {
            char message[VPI_MAX_STATUS_MESSAGE_LENGTH]{};
            vpiGetLastStatusMessage(message, sizeof(message));

            std::cerr << "Failed to rebind VPI image wrapper: "
                      << vpiStatusGetName(status)
                      << " - " << message << '\n';

            return false;
        }
        return true;
    }

    void ImageWrapper::release() noexcept {
        if (image_ != nullptr){
            vpiImageDestroy(image_);
            image_ = nullptr;
        }
    }
}
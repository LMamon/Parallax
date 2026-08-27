#include <parallax/isp/isp.hpp>

#include <parallax/isp/demosaic.cuh>
#include <linux/videodev2.h>

namespace parallax::isp {

    ISP::ISP() = default;
    ISP::~ISP() { shutdown(); }

    bool ISP::initialize(const parallax::camera::CameraConfig& config) {
        if (initialized_) return true;

        // Allocate the GPU Bayer input buffer.
        if (!gpu_input_.buffer.allocate(config.width, config.height, 1, sizeof(std::uint16_t))) {
            return false;
        }

        gpu_input_.width = config.width;
        gpu_input_.height = config.height;
        gpu_input_.pattern = config.bayer_pattern;

        // For now the camera always produces BA10/GRBG.
        // Later this should come from an ISPConfig or CameraConfig
        // once multiple sensor formats are supported.
        // gpu_input_.fourcc = device_->getPixelFormat();
        const std::uint32_t stereo_width = config.width / 2;

        if (!output_pool_.initialize([&](OutputSlot& slot, std::size_t index) {
                    slot.rgb.width = stereo_width;
                    slot.rgb.height = config.height;
                    slot.rgb.storage_slot = static_cast<std::uint32_t>(index);

                    slot.gray.width = stereo_width;
                    slot.gray.height = config.height;
                    slot.gray.storage_slot = static_cast<std::uint32_t>(index);

                    if (!slot.rgb.left.allocate(stereo_width,
                                                config.height,
                                                StereoRgbFrame::Channels,
                                                sizeof(std::uint8_t)) ||
                        !slot.rgb.right.allocate(stereo_width,
                                                 config.height,
                                                 StereoRgbFrame::Channels,
                                                 sizeof(std::uint8_t))) {

                        return false;
                    }

                    if (!slot.gray.left.allocate(stereo_width,
                                                 config.height,
                                                 StereoGrayFrame::Channels,
                                                 sizeof(std::uint8_t)) ||
                        !slot.gray.right.allocate(stereo_width,
                                                  config.height,
                                                  StereoGrayFrame::Channels,
                                                  sizeof(std::uint8_t))) {

                        return false;
                    }

                    return true;
                })) {

            shutdown();
            return false;
        }

        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            shutdown();
            return false;
        }

        initialized_ = true;
        return true;
    }

    bool ISP::downloadRaw(std::uint16_t* host_data, std::size_t host_pitch) const {
        if (!initialized_ || host_data == nullptr) return false;

        return gpu_input_.buffer.downloadAsync(host_data, host_pitch, stream_);
    }

    bool ISP::process(const parallax::camera::RawFrame& input, OutputSlot& output) {
        if (!initialized_) return false;
        if (!upload(input)) return false;

        return demosaicAndSplit(gpu_input_, output.rgb, output.gray, stream_);
    }

    bool ISP::upload(const parallax::camera::RawFrame& input) {
        return gpu_input_.buffer.uploadAsync(input.data,
                                            input.width * sizeof(std::uint16_t),
                                            stream_);
    }

    void ISP::shutdown() {
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }

        gpu_input_.buffer.release();
        output_pool_.reset();

        initialized_ = false;
    }

    bool ISP::synchronize() {
        if (!initialized_ || stream_ == nullptr) return false;

        return cudaStreamSynchronize(stream_) == cudaSuccess;
    }
}
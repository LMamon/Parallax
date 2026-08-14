#include <parallax/stereo/rectification.hpp>

#include <vpi/Stream.h>
#include <vpi/WarpMap.h>
#include <vpi/algo/Remap.h>

#include <cstdint>
#include <iostream>

namespace parallax::stereo {

    namespace {
        void logVpiError(const char* message, VPIStatus status) {
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH]{};
            vpiGetLastStatusMessage(buffer, sizeof(buffer));

            std::cerr 
                << message << ": "
                << vpiStatusGetName(status) << " - "
                << buffer << '\n';
        }

        bool populateWarpMap(VPIWarpMap& warp, 
                            std::uint32_t width, 
                            std::uint32_t height, 
                            const std::vector<float>& map_x,
                            const std::vector<float>& map_y) {

            const std::size_t expected_elements = static_cast<std::size_t>(width) * 
                                                static_cast<std::size_t>(height);

            if (map_x.size() != expected_elements || map_y.size() != expected_elements) {
                std::cerr
                    << "Invalid rectification map dimensions\n"
                    << "Expected elements: " << expected_elements
                    << "\nmap_x: " << map_x.size()
                    << "\nmap_y: " << map_y.size()
                    << '\n';

                return false;
            }

            warp = {};
            warp.grid.numHorizRegions = 1;
            warp.grid.numVertRegions = 1;

            warp.grid.horizInterval[0] = 1;
            warp.grid.vertInterval[0] = 1;

            warp.grid.regionWidth[0] = static_cast<int16_t>(width);
            warp.grid.regionHeight[0] = static_cast<int16_t>(height);
            
            VPIStatus status = vpiWarpMapAllocData(&warp);
            if (status != VPI_SUCCESS) {
                std::cerr << "Failed to allocate VPI warp map\n";
                return false;
            }

            for (std::uint32_t y = 0; y < height; ++y) {
                auto* row = reinterpret_cast<VPIKeypointF32*>(reinterpret_cast<std::uint8_t*>(warp.keypoints) +
                                                                static_cast<std::size_t>(y) * warp.pitchBytes);
                
                for (std::uint32_t x = 0; x < width; ++x) {
                    const std::size_t i = static_cast<std::size_t>(y) * width + x;

                    row[x].x = map_x[i];
                    row[x].y = map_y[i];
                }
            }
            return true;
            
        }
    }

        StereoRectifier::~StereoRectifier() { shutdown(); }

        bool StereoRectifier::initialize(const StereoCalibration& calibration, 
                                        const parallax::isp::StereoRgbFrame& rgb_input,
                                        const parallax::isp::StereoGrayFrame& gray_input,
                                        VPIStream stream) {

            if (initialized_) return true;
            if (!calibration.loaded()) {
                std::cerr << "Stereo calibration is not loaded\n";
                return false;
            }

            if (stream == nullptr) {
                std::cerr << "StereoRectifier received null VPI stream\n";
                return false;
            }

            // VPI iniitialization goes here next.
            const auto& metadata = calibration.metadata();

            if (rgb_input.width != metadata.image_width || rgb_input.height != metadata.image_height) {
                std::cerr
                    << "Stereo input dimensions do not match calibration\nInput: "
                    << rgb_input.width << 'x' << rgb_input.height
                    << "\nCalibration: "
                    << metadata.image_width << 'x'
                    << metadata.image_height << '\n';

                return false;
            }
            if (gray_input.width != metadata.image_width || gray_input.height != metadata.image_height) {
                std::cerr
                    << "Stereo grayscale input dimensions do not match calibration\n"
                    << "Input: "
                    << gray_input.width << 'x' << gray_input.height
                    << "\nCalibration: "
                    << metadata.image_width << 'x'
                    << metadata.image_height << '\n';

                return false;
            }

            if (!rgb_input.left.isAllocated() || !rgb_input.right.isAllocated()) {
                std::cerr << "ISP stereo RGB buffers are not allocated\n";
                return false;
            }
            if (!gray_input.left.isAllocated() || !gray_input.right.isAllocated()) {
                std::cerr << "ISP stereo grayscale buffers are not allocated\n";
                return false;
            }
            stream_ = stream;

            rgb_output_.width = rgb_input.width;
            rgb_output_.height = rgb_input.height;

            gray_output_.width = gray_input.width;
            gray_output_.height = gray_input.height;

            // allocate gpu-resident output images
            if (!rgb_output_.left.allocate(rgb_input.width, rgb_input.height, parallax::isp::RectifiedStereoFrame::Channels, sizeof(std::uint8_t)) ||
                !rgb_output_.right.allocate(rgb_input.width, rgb_input.height, parallax::isp::RectifiedStereoFrame::Channels, sizeof(std::uint8_t))) {

                std::cerr << "Failed to allocated rectified stereo buffers\n";
                shutdown();
                return false;
            }
            if (!gray_output_.left.allocate(gray_input.width, 
                                            gray_input.height, 
                                            parallax::isp::RectifiedStereoGrayFrame::Channels, 
                                            sizeof(std::uint8_t)) ||

                !gray_output_.right.allocate(gray_input.width, 
                                            gray_input.height, 
                                            parallax::isp::RectifiedStereoGrayFrame::Channels, 
                                            sizeof(std::uint8_t))) {

                std::cerr << "Failed to allocate rectified grayscale buffers\n";
                shutdown();
                return false;
            }
            // wrap existing CUDA allocations for VPI
            // input.left/right are owned by ISP.
            // output_.left/right are owned by this StereoRectifier
            // RGB
            if (!rgb_left_input_.create(rgb_input.left, VPI_IMAGE_FORMAT_RGB8) ||
                !rgb_right_input_.create(rgb_input.right, VPI_IMAGE_FORMAT_RGB8) ||
                !rgb_left_output_.create(rgb_output_.left, VPI_IMAGE_FORMAT_RGB8) ||
                !rgb_right_output_.create(rgb_output_.right, VPI_IMAGE_FORMAT_RGB8)) {

                std::cerr << "Failed to create RGB VPI image wrappers\n";
                shutdown();
                return false;
            }

            // Grayscale
            if (!gray_left_input_.create(gray_input.left, VPI_IMAGE_FORMAT_Y8_ER) ||
                !gray_right_input_.create(gray_input.right, VPI_IMAGE_FORMAT_Y8_ER) ||
                !gray_left_output_.create(gray_output_.left, VPI_IMAGE_FORMAT_Y8_ER) ||
                !gray_right_output_.create(gray_output_.right, VPI_IMAGE_FORMAT_Y8_ER)) {

                std::cerr << "Failed to create grayscale VPI image wrappers\n";
                shutdown();
                return false;
            }

            if (!populateWarpMap(left_warp_, rgb_input.width, rgb_input.height, calibration.leftMapX(), calibration.leftMapY())) {
                shutdown();
                return false;
            }
            left_warp_allocated_ = true;

            if (!populateWarpMap(right_warp_, rgb_input.width, rgb_input.height, calibration.rightMapX(), calibration.rightMapY())) {
                shutdown();
                return false;
            }
            right_warp_allocated_ = true;

            VPIStatus status = vpiCreateRemap(VPI_BACKEND_CUDA, &left_warp_, &left_remap_);

            if (status != VPI_SUCCESS) {
                logVpiError("Failed to create left VPI remap payload", status);
                shutdown();
                return false;
            }

            status = vpiCreateRemap(VPI_BACKEND_CUDA, &right_warp_, &right_remap_);

            if (status != VPI_SUCCESS) {
                logVpiError("Failed to create right VPI remap payload", status);
                shutdown();
                return false;
            }

            initialized_ = true;
            return true;
        }

        bool StereoRectifier::process() {
            if (!initialized_) return false;

            VPIStatus status = vpiSubmitRemap(stream_,
                                            VPI_BACKEND_CUDA,
                                            left_remap_,
                                            rgb_left_input_.handle(),
                                            rgb_left_output_.handle(),
                                            VPI_INTERP_LINEAR,
                                            VPI_BORDER_ZERO,
                                            0);

            if (status != VPI_SUCCESS) {
                logVpiError("Failed to submit left RGB remap", status);
                return false;
            }
            status = vpiSubmitRemap(stream_,
                                    VPI_BACKEND_CUDA,
                                    right_remap_,
                                    rgb_right_input_.handle(),
                                    rgb_right_output_.handle(),
                                    VPI_INTERP_LINEAR,
                                    VPI_BORDER_ZERO,
                                    0);

            if (status != VPI_SUCCESS) {
                logVpiError("Failed to submit right RGB remap", status);
                return false;
            }

            status = vpiSubmitRemap(stream_,
                                    VPI_BACKEND_CUDA,
                                    left_remap_,
                                    gray_left_input_.handle(),
                                    gray_left_output_.handle(),
                                    VPI_INTERP_LINEAR,
                                    VPI_BORDER_ZERO,
                                    0);

            if (status != VPI_SUCCESS) {
                logVpiError("Failed to submit left grayscale remap", status);
                return false;
            }

            status = vpiSubmitRemap(stream_,
                                    VPI_BACKEND_CUDA,
                                    right_remap_,
                                    gray_right_input_.handle(),
                                    gray_right_output_.handle(),
                                    VPI_INTERP_LINEAR,
                                    VPI_BORDER_ZERO,
                                    0);

            if (status != VPI_SUCCESS) {
                logVpiError("Failed to submit right grayscale remap", status);
                return false;
            }
            return true;
        }

        bool StereoRectifier::synchronize() {
            if (!initialized_) return false;
            return vpiStreamSync(stream_) == VPI_SUCCESS;
        }

        void StereoRectifier::shutdown() {
            if (left_remap_ != nullptr) {
                vpiPayloadDestroy(left_remap_);
                left_remap_ = nullptr;
            }

            if (right_remap_ != nullptr) {
                vpiPayloadDestroy(right_remap_);
                right_remap_ = nullptr;
            }

            if (left_warp_allocated_) {
                vpiWarpMapFreeData(&left_warp_);
                left_warp_allocated_ = false;
            }

            if (right_warp_allocated_) {
                vpiWarpMapFreeData(&right_warp_);
                right_warp_allocated_ = false;
            }

            rgb_left_input_.release();
            rgb_right_input_.release();
            rgb_left_output_.release();
            rgb_right_output_.release();

            gray_left_input_.release();
            gray_right_input_.release();
            gray_left_output_.release();
            gray_right_output_.release();

            rgb_output_.left.release();
            rgb_output_.right.release();

            gray_output_.left.release();
            gray_output_.right.release();

            rgb_output_.width = 0;
            rgb_output_.height = 0;

            gray_output_.width = 0;
            gray_output_.height = 0;

            stream_ = nullptr;
            initialized_ = false;
        }
}
#include <parallax/stereo/matcher.hpp>

#include <vpi/Status.h>
#include <vpi/algo/StereoDisparity.h>

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
    }

    StereoMatcher::~StereoMatcher() { shutdown(); }

    bool StereoMatcher::initialize(const parallax::isp::RectifiedStereoGrayFrame& input, VPIStream stream) {
        if (initialized_) return true;
        if (stream == nullptr) {
            std::cerr << "StereoMatcher received null VPI stream\n";
            return false;
        }

        if (!input.left.isAllocated() || !input.right.isAllocated()) {
            std::cerr << "Rectified stereo buffers are not allocated\n";
            return false;
        }

        if (input.width == 0 || input.height == 0) {
            std::cerr << "Invalid rectified stereo dimensions\n";
            return false;
        }

        stream_ = stream;
        output_.width = input.width;
        output_.height = input.height;

        if (!output_.disparity.allocate(input.width, input.height, 1, sizeof(std::int16_t))) {
            std::cerr << "Failed to allocate disparity buffer\n";
            shutdown();
            return false;
        }

        if (!output_.confidence.allocate(input.width, input.height, 1, sizeof(std::uint16_t))) {
            std::cerr << "Failed to allocate confidence buffer\n";
            shutdown();
            return false;
        }
        
        // Wrap all CUDA allocations without copying
        if (!left_input_.create(input.left, VPI_IMAGE_FORMAT_Y8_ER) ||
            !right_input_.create(input.right, VPI_IMAGE_FORMAT_Y8_ER) ||
            !disparity_image_.create(output_.disparity, VPI_IMAGE_FORMAT_S16) ||
            !confidence_image_.create(output_.confidence, VPI_IMAGE_FORMAT_U16)) {

            std::cerr << "Failed to create StereoMatcher VPI wrappers\n";
            shutdown();
            return false;
        }

        // Stereo disparity estimator creation parameters.
        VPIStereoDisparityEstimatorCreationParams create_params{};
        VPIStatus status = vpiInitStereoDisparityEstimatorCreationParams(&create_params);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to initialize stereo disparity creation parameters", status);

            shutdown();
            return false;
        }

        // Start with the full CUDA-supported search range.
        // Tune later based on the working distance of the rig.
        create_params.maxDisparity = 128; //256 too large for Jetson Orin NanoSDK
        create_params.downscaleFactor = 1;
        // 

        status = vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA,
                                                    static_cast<int32_t>(input.width),
                                                    static_cast<int32_t>(input.height),
                                                    VPI_IMAGE_FORMAT_Y8_ER,
                                                    &create_params,
                                                    &stereo_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create VPI stereo disparity estimator", status);
            shutdown();
            return false;
        }

        // Runtime submission parameters.
        status = vpiInitStereoDisparityEstimatorParams(&submit_params_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to initialize stereo disparity parameters", status);
            shutdown();
            return false;
        }

        initialized_ = true;
        return true;
    }

    bool StereoMatcher::process() {
        if (!initialized_) return false;
        // Because this uses the same stream as StereoRectifier,
        // VPI guarantees ordering after the preceding remap work
        VPIStatus status = vpiSubmitStereoDisparityEstimator(stream_,
                                                            VPI_BACKEND_CUDA,
                                                            stereo_,
                                                            left_input_.handle(),
                                                            right_input_.handle(),
                                                            disparity_image_.handle(),
                                                            confidence_image_.handle(),
                                                            &submit_params_);

            if (status != VPI_SUCCESS) {
                logVpiError("Failed to submit VPI stereo disparity estimator", status);
                return false;
            }
        return true;
    }

    void StereoMatcher::shutdown() {
        // Shared stream is NOT synchronized/destroyed here.
        // Caller owns its lifecycle.
        if (stereo_ != nullptr) {
            vpiPayloadDestroy(stereo_);
            stereo_ = nullptr;
        }

        left_input_.release();
        right_input_.release();

        disparity_image_.release();
        confidence_image_.release();

        output_.disparity.release();
        output_.confidence.release();

        output_.width = 0;
        output_.height = 0;

        stream_ = nullptr;
        initialized_ = false;
    }
}
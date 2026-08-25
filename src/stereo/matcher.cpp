#include <parallax/stereo/matcher.hpp>

#include <vpi/Status.h>
#include <vpi/algo/StereoDisparity.h>
#include <vpi/Image.h>
#include <vpi/algo/ConvertImageFormat.h>

#include <cstdint>
#include <iostream>
#include <chrono>
#include <iomanip>

namespace parallax::stereo {
    namespace {
        void logVpiError(const char* message, VPIStatus status) {
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH]{};
            vpiGetLastStatusMessage(buffer, sizeof(buffer));

            std::cerr << message << ": "
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

        // if (!output_.confidence.allocate(input.width, input.height, 1, sizeof(std::uint16_t))) {
        //     std::cerr << "Failed to allocate confidence buffer\n";
        //     shutdown();
        //     return false;
        // }
        
        // Wrap all CUDA allocations without copying
        if (!left_input_.create(input.left, VPI_IMAGE_FORMAT_Y8_ER) ||
            !right_input_.create(input.right, VPI_IMAGE_FORMAT_Y8_ER) ||
            !disparity_image_.create(output_.disparity, VPI_IMAGE_FORMAT_S16)) {

            std::cerr << "Failed to create StereoMatcher VPI wrappers\n";
            shutdown();
            return false;
        }
        // OFA in VPI 3.2 requires block-linear stereo input.
        //
        // Rectification intentionally remains pitch-linear Y8_ER because VPI Remap
        // requires input/output format equality. VIC performs the required layout
        // transition here without CPU staging.
        VPIStatus status = vpiImageCreate(static_cast<int32_t>(input.width),
                                          static_cast<int32_t>(input.height),
                                          VPI_IMAGE_FORMAT_Y8_ER_BL,
                                          VPI_BACKEND_VIC | VPI_BACKEND_OFA,
                                          &left_block_linear_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create left block-linear image", status);
            shutdown();
            return false;
        }

        // OFA in VPI 3.2 requires block-linear stereo input.
        //
        // Rectification intentionally remains pitch-linear Y8_ER because VPI Remap
        // requires input/output format equality. VIC performs the required layout
        // transition here without CPU staging.
        status = vpiImageCreate(static_cast<int32_t>(input.width),
                                static_cast<int32_t>(input.height),
                                VPI_IMAGE_FORMAT_Y8_ER_BL,
                                VPI_BACKEND_VIC | VPI_BACKEND_OFA,
                                &right_block_linear_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create right block-linear image", status);
            shutdown();
            return false;
        }


        // OFA-only stereo requires block-linear S16 disparity. Values are Q10.5
        // fixed point and retain the existing StereoMatchFrame::DisparityScale = 32.
        status = vpiImageCreate(static_cast<int32_t>(input.width),
                                static_cast<int32_t>(input.height),
                                VPI_IMAGE_FORMAT_S16_BL,
                                VPI_BACKEND_OFA | VPI_BACKEND_VIC,
                                &disparity_block_linear_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create block-linear disparity image", status);
            shutdown();
            return false;
        }

        // Stereo disparity estimator creation parameters.
        VPIStereoDisparityEstimatorCreationParams create_params{};
        status = vpiInitStereoDisparityEstimatorCreationParams(&create_params);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to initialize stereo disparity creation parameters", status);
            shutdown();
            return false;
        }

        // Start with the full CUDA-supported search range.
        // Tune later based on the working distance of the rig.
        create_params.maxDisparity = 128; //256 too large for Jetson Orin NanoSDK
        create_params.downscaleFactor = 1;
        create_params.includeDiagonals = 1;
        // 

        status = vpiCreateStereoDisparityEstimator(VPI_BACKEND_OFA,
                                                   static_cast<int32_t>(input.width),
                                                   static_cast<int32_t>(input.height),
                                                   VPI_IMAGE_FORMAT_Y8_ER_BL,
                                                   &create_params,
                                                   &stereo_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create OFA stereo disparity estimator", status);
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

        submit_params_.numPasses = 3;
        initialized_ = true;
        return true;
    }

    bool StereoMatcher::process() {
        if (!initialized_) return false;

        VPIStatus status;

        status = vpiSubmitConvertImageFormat(stream_,
                                            VPI_BACKEND_VIC,
                                            left_input_.handle(),
                                            left_block_linear_,
                                            nullptr);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to convert left image to block-linear", status);
            return false;
        }

        status = vpiSubmitConvertImageFormat(stream_,
                                            VPI_BACKEND_VIC,
                                            right_input_.handle(),
                                            right_block_linear_,
                                            nullptr);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to convert right image to block-linear", status);
            return false;
        }

        status = vpiSubmitStereoDisparityEstimator(stream_,
                                                   VPI_BACKEND_OFA,
                                                   stereo_,
                                                   left_block_linear_,
                                                   right_block_linear_,
                                                   disparity_block_linear_,
                                                   nullptr,
                                                   &submit_params_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to submit OFA stereo disparity estimator", status);
            return false;
        }

        status = vpiSubmitConvertImageFormat(stream_,
                                             VPI_BACKEND_VIC,
                                             disparity_block_linear_,
                                             disparity_image_.handle(),
                                             nullptr);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to convert disparity to pitch-linear", status);
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

        if (left_block_linear_ != nullptr) {
            vpiImageDestroy(left_block_linear_);
            left_block_linear_ = nullptr;
        }

        if (right_block_linear_ != nullptr) {
            vpiImageDestroy(right_block_linear_);
            right_block_linear_ = nullptr;
        }

        if (disparity_block_linear_ != nullptr) {
            vpiImageDestroy(disparity_block_linear_);
            disparity_block_linear_ = nullptr;
        }

        left_input_.release();
        right_input_.release();

        disparity_image_.release();

        output_.disparity.release();
        // output_.confidence.release();

        output_.width = 0;
        output_.height = 0;

        stream_ = nullptr;
        initialized_ = false;
    }
}
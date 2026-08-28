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
        // OFA in VPI 3.2 requires block-linear stereo input.
        //
        // Rectification intentionally remains pitch-linear Y8_ER because VPI Remap
        // requires input/output format equality. VIC performs the required layout
        // transition here without CPU staging.
        VPIStatus status;
        if (!output_pool_.initialize([&](OutputSlot& slot, std::size_t index) {

                    slot.output.width = input.width;
                    slot.output.height = input.height;
                    slot.output.storage_slot = static_cast<std::uint32_t>(index);

                    if (!slot.output.disparity.allocate(input.width,
                                                        input.height,
                                                        1,
                                                        sizeof(std::int16_t))) {

                        return false;
                    }

                    if (!slot.disparity_image.create(slot.output.disparity, VPI_IMAGE_FORMAT_S16)) {
                        return false;
                    }

                    status = vpiImageCreate(static_cast<int32_t>(input.width),
                                                     static_cast<int32_t>(input.height),
                                                     VPI_IMAGE_FORMAT_Y8_ER_BL,
                                                     VPI_BACKEND_VIC | VPI_BACKEND_OFA,
                                                     &slot.left_block_linear);

                    if (status != VPI_SUCCESS) {
                        logVpiError("Failed to create left block-linear image", status);
                        return false;
                    }

                    status = vpiImageCreate(static_cast<int32_t>(input.width),
                                            static_cast<int32_t>(input.height),
                                            VPI_IMAGE_FORMAT_Y8_ER_BL,
                                            VPI_BACKEND_VIC | VPI_BACKEND_OFA,
                                            &slot.right_block_linear);

                    if (status != VPI_SUCCESS) {
                        logVpiError("Failed to create right block-linear image", status);
                        return false;
                    }

                    status = vpiImageCreate(static_cast<int32_t>(input.width),
                                            static_cast<int32_t>(input.height),
                                            VPI_IMAGE_FORMAT_S16_BL,
                                            VPI_BACKEND_OFA | VPI_BACKEND_VIC,
                                            &slot.disparity_block_linear);

                    if (status != VPI_SUCCESS) {
                        logVpiError("Failed to create block-linear disparity image", status);
                        return false;
                    }

                    return true;
                })) {

            std::cerr << "Failed to initialize StereoMatcher output pool\n";
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

    bool StereoMatcher::process(const parallax::isp::RectifiedStereoGrayFrame& input,
                                OutputSlot& output,
                                VPIStream stream) {

        if (!initialized_ || stream == nullptr) {
            return false;
        }

        /**
         * Rectification output is now rotating pooled storage. Rebind the input
         * wrappers to the exact RectifiedGray generation consumed by this
         * submission.
         */
        if (!output.left_input.rebind(input.left, VPI_IMAGE_FORMAT_Y8_ER) ||
            !output.right_input.rebind(input.right, VPI_IMAGE_FORMAT_Y8_ER)) {
            return false;
        }

        VPIStatus status = vpiSubmitConvertImageFormat(stream,
                                                       VPI_BACKEND_VIC,
                                                       output.left_input.handle(),
                                                       output.left_block_linear,
                                                       nullptr);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to convert left image to block-linear", status);
            return false;
        }

        status = vpiSubmitConvertImageFormat(stream,
                                             VPI_BACKEND_VIC,
                                             output.right_input.handle(),
                                             output.right_block_linear,
                                             nullptr);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to convert right image to block-linear", status);
            return false;
        }

        status = vpiSubmitStereoDisparityEstimator(stream,
                                                  VPI_BACKEND_OFA,
                                                  stereo_,
                                                  output.left_block_linear,
                                                  output.right_block_linear,
                                                  output.disparity_block_linear,
                                                  nullptr,
                                                  &submit_params_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to submit OFA stereo disparity estimator", status);
            return false;
        }

        status = vpiSubmitConvertImageFormat(stream,
                                            VPI_BACKEND_VIC,
                                            output.disparity_block_linear,
                                            output.disparity_image.handle(),
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
        
        output_pool_.reset();

        stream_ = nullptr;
        initialized_ = false;
    }
}
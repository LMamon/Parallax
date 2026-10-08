#include <parallax/stereo/matcher.hpp>
#include <parallax/core/execution_context.hpp>

#include <vpi/Status.h>

#include <cstdint>
#include <iostream>

namespace parallax::stereo {
namespace {

constexpr std::uint64_t kStereoBackend = VPI_BACKEND_CUDA;
constexpr int kMaxDisparity = 128;

void logVpiError(const char* message, VPIStatus status) {
    char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH]{};
    vpiGetLastStatusMessage(buffer, sizeof(buffer));
    std::cerr << message << ": " << vpiStatusGetName(status)
              << " - " << buffer << '\n';
}

}  // namespace

StereoMatcher::~StereoMatcher() { shutdown(); }

bool StereoMatcher::initialize(
    const parallax::isp::RectifiedStereoGrayFrame& input,
    VPIStream stream) {
    if (initialized_) return true;
    if (stream == nullptr || input.width == 0 || input.height == 0 ||
        !input.left.isAllocated() || !input.right.isAllocated()) {
        return false;
    }

    stream_ = stream;

    if (!output_pool_.initialize([&](OutputSlot& slot, std::size_t index) {
            slot.output.width = input.width;
            slot.output.height = input.height;
            slot.output.storage_slot = static_cast<std::uint32_t>(index);

            if (!slot.output.disparity.allocate(
                    input.width, input.height, 1, sizeof(std::int16_t)) ||
                !slot.output.confidence.allocate(
                    input.width, input.height, 1, sizeof(std::uint16_t))) {
                return false;
            }
            return slot.disparity_image.create(
                       slot.output.disparity, VPI_IMAGE_FORMAT_S16) &&
                   slot.confidence_image.create(
                       slot.output.confidence, VPI_IMAGE_FORMAT_U16);
        })) {
        shutdown();
        return false;
    }

    VPIStereoDisparityEstimatorCreationParams create_params{};
    VPIStatus status =
        vpiInitStereoDisparityEstimatorCreationParams(&create_params);
    if (status != VPI_SUCCESS) {
        logVpiError("Failed to initialize CUDA SGM creation parameters", status);
        shutdown();
        return false;
    }
    create_params.maxDisparity = kMaxDisparity;

    status = vpiCreateStereoDisparityEstimator(
        kStereoBackend,
        static_cast<int32_t>(input.width),
        static_cast<int32_t>(input.height),
        VPI_IMAGE_FORMAT_Y8_ER,
        &create_params,
        &stereo_);
    if (status != VPI_SUCCESS) {
        logVpiError("Failed to create CUDA stereo disparity estimator", status);
        shutdown();
        return false;
    }

    status = vpiInitStereoDisparityEstimatorParams(&submit_params_);
    if (status != VPI_SUCCESS) {
        logVpiError("Failed to initialize CUDA SGM parameters", status);
        shutdown();
        return false;
    }

    // Keep VPI defaults from the isolated benchmark; only the payload-bound
    // disparity range is explicit.
    submit_params_.maxDisparity = kMaxDisparity;
    initialized_ = true;
    return true;
}

std::shared_ptr<StereoMatcher::OutputSlot> StereoMatcher::acquireOutput(
    parallax::core::ExecutionContext& context) {
    auto output = output_pool_.acquire();
    if (!output) return {};
    if (output->completion.valid()) {
        if (!context.waitForHost(output->completion)) return {};
        output->completion = {};
    }
    return output;
}

std::shared_ptr<StereoMatcher::OutputSlot>
StereoMatcher::acquireOutputSynchronous() {
    auto output = output_pool_.acquire();
    if (!output) return {};

    // The synchronous compatibility path must never inherit an outstanding
    // asynchronous completion from a graph-managed use of this slot.
    if (output->completion.valid()) {
        return {};
    }

    return output;
}


bool StereoMatcher::process(
    const parallax::isp::RectifiedStereoGrayFrame& input,
    OutputSlot& output,
    VPIStream stream) {
    if (!initialized_ || stream == nullptr) return false;

    if (!output.left_input.rebind(input.left, VPI_IMAGE_FORMAT_Y8_ER) ||
        !output.right_input.rebind(input.right, VPI_IMAGE_FORMAT_Y8_ER)) {
        return false;
    }

    const VPIStatus status = vpiSubmitStereoDisparityEstimator(
        stream,
        kStereoBackend,
        stereo_,
        output.left_input.handle(),
        output.right_input.handle(),
        output.disparity_image.handle(),
        output.confidence_image.handle(),
        &submit_params_);
    if (status != VPI_SUCCESS) {
        logVpiError("Failed to submit CUDA stereo disparity estimator", status);
        return false;
    }
    return true;
}

void StereoMatcher::shutdown() {
    if (stereo_ != nullptr) {
        vpiPayloadDestroy(stereo_);
        stereo_ = nullptr;
    }
    output_pool_.reset();
    stream_ = nullptr;
    initialized_ = false;
}

}  // namespace parallax::stereo

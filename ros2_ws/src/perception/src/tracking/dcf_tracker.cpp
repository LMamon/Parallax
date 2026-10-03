#include <parallax/tracking/dcf_tracker.hpp>

#include <algorithm>
#include <iostream>
#include <utility>

#include <vpi/Status.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/CropScaler.h>
#include <vpi/algo/DCFTracker.h>

namespace parallax::tracking {
    namespace {
        constexpr int32_t kMaxSequences = 1;
        constexpr int32_t kMaxObjects = 5;
    }

    DcfTracker::~DcfTracker() { reset(); }

    void DcfTracker::logVpiError(const char* message, VPIStatus status) const {
        char detail[VPI_MAX_STATUS_MESSAGE_LENGTH]{};
        vpiGetLastStatusMessage(detail, sizeof(detail));

        std::cerr << message << ": " << vpiStatusGetName(status) << " - " << detail << '\n';
    }

    bool DcfTracker::ensureResources(const parallax::cuda::CudaBuffer& image) {
        if (!image.isAllocated() ||
            image.channels() != 3 ||
            image.elementSize() != sizeof(std::uint8_t)) {
            return false;
        }

        if (stream_ != nullptr && width_ == image.width() && height_ == image.height()) {
            return true;
        }

        reset();

        width_ = image.width();
        height_ = image.height();

        VPIStatus status = vpiInitDCFTrackerCreationParams(&creation_params_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to initialize DCF creation parameters", status);
            reset();
            return false;
        }

        status = vpiInitDCFTrackerParams(&params_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to initialize DCF parameters", status);
            reset();
            return false;
        }

        patch_size_ = creation_params_.featurePatchSize * creation_params_.hogCellSize;

        status = vpiStreamCreate(0, &stream_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create DCF VPI stream", status);
            reset();
            return false;
        }

        status = vpiCreateCropScaler(VPI_BACKEND_CUDA, kMaxSequences, kMaxObjects, &crop_scaler_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create DCF crop scaler", status);
            reset();
            return false;
        }

        status = vpiCreateDCFTracker(VPI_BACKEND_CUDA, kMaxSequences, kMaxObjects, &creation_params_, &dcf_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create DCF tracker", status);
            reset();
            return false;
        }

        status = vpiArrayCreate(kMaxObjects, VPI_ARRAY_TYPE_DCF_TRACKED_BOUNDING_BOX, VPI_BACKEND_CPU | VPI_BACKEND_CUDA, &in_targets_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create DCF input targets", status);
            reset();
            return false;
        }

        status = vpiArrayCreate(kMaxObjects, VPI_ARRAY_TYPE_DCF_TRACKED_BOUNDING_BOX, VPI_BACKEND_CPU | VPI_BACKEND_CUDA, &out_targets_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create DCF output targets", status);
            reset();
            return false;
        }

        status = vpiArrayCreate(kMaxObjects, VPI_ARRAY_TYPE_F32, VPI_BACKEND_CPU | VPI_BACKEND_CUDA, &max_responses_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create DCF response array", status);
            reset();
            return false;
        }

        // CropScaler needs RGBA8 input, so keep this conversion device-side and persistent.
        status = vpiImageCreate(static_cast<int32_t>(width_), 
                                static_cast<int32_t>(height_), 
                                VPI_IMAGE_FORMAT_RGBA8, 
                                VPI_BACKEND_CUDA, 
                                &rgba_frame_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create DCF RGBA frame", status);
            reset();
            return false;
        }

        status = vpiImageCreate(patch_size_, patch_size_ * kMaxObjects, VPI_IMAGE_FORMAT_RGBA8, VPI_BACKEND_CUDA, &target_patch_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to create DCF target patch", status);
            reset();
            return false;
        }
        return true;
    }

    bool DcfTracker::bindInput(const parallax::cuda::CudaBuffer& image) {
        if (!input_wrapper_.valid()) {
            return input_wrapper_.create(image, VPI_IMAGE_FORMAT_RGB8);
        }
        return input_wrapper_.rebind(image, VPI_IMAGE_FORMAT_RGB8);
    }

    bool DcfTracker::convertInput() {
        const VPIStatus status = vpiSubmitConvertImageFormat(stream_,
                                                             VPI_BACKEND_CUDA,
                                                             input_wrapper_.handle(),
                                                             rgba_frame_,
                                                             nullptr);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to convert DCF input to RGBA8", status);
            return false;
        }

        return true;
    }

    bool DcfTracker::crop(VPIArray objects) {
        VPIImage frame = rgba_frame_;

        const VPIStatus status = vpiSubmitCropScalerBatch(stream_,
                                                          VPI_BACKEND_CUDA,
                                                          crop_scaler_,
                                                          &frame,
                                                          1,
                                                          objects,
                                                          patch_size_,
                                                          patch_size_,
                                                          target_patch_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to crop DCF target patch", status);
            return false;
        }

        return true;
    }

    bool DcfTracker::sync() {
        const VPIStatus status = vpiStreamSync(stream_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to synchronize DCF stream", status);
            return false;
        }

        return true;
    }

    bool DcfTracker::initialize(const parallax::cuda::CudaBuffer& image, const cv::Rect2f& box) {
        initialized_ = false;

        if (box.width <= 0.0F || box.height <= 0.0F) {
            return false;
        }

        if (!ensureResources(image) || !bindInput(image)) {
            return false;
        }

        VPIArrayData targets{};
        VPIStatus status = vpiArrayLockData(in_targets_, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &targets);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to lock DCF input target", status);
            return false;
        }

        auto* target = static_cast<VPIDCFTrackedBoundingBox*>(targets.buffer.aos.data);

        target[0] = {};
        target[0].bbox.left = box.x;
        target[0].bbox.top = box.y;
        target[0].bbox.width = box.width;
        target[0].bbox.height = box.height;
        target[0].state = VPI_TRACKING_STATE_NEW;
        target[0].seqIndex = 0;
        target[0].filterLR = 0.075F;
        target[0].filterChannelWeightsLR = 0.1F;

        *targets.buffer.aos.sizePointer = 1;

        status = vpiArrayUnlock(in_targets_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to unlock DCF input target", status);
            return false;
        }

        if (!convertInput() || !crop(in_targets_)) {
            return false;
        }

        status = vpiSubmitDCFTrackerUpdateBatch(stream_,
                                                VPI_BACKEND_CUDA,
                                                dcf_,
                                                nullptr,
                                                0,
                                                nullptr,
                                                nullptr,
                                                target_patch_,
                                                in_targets_,
                                                &params_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to initialize DCF target", status);
            return false;
        }

        if (!sync()) return false;
        initialized_ = true;
        return true;
    }

    DcfTrackerResult DcfTracker::update(const parallax::cuda::CudaBuffer& image) {

        DcfTrackerResult result{};

        if (!initialized_ ||
            image.width() != width_ ||
            image.height() != height_ ||
            !bindInput(image) ||
            !convertInput() ||
            !crop(in_targets_)) {
            return result;
        }

        VPIStatus status = vpiSubmitDCFTrackerLocalizeBatch(stream_,
                                                            VPI_BACKEND_CUDA,
                                                            dcf_,
                                                            nullptr,
                                                            0,
                                                            nullptr,
                                                            target_patch_,
                                                            in_targets_,
                                                            out_targets_,
                                                            nullptr,
                                                            max_responses_,
                                                            &params_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to localize DCF target", status);
            return result;
        }

        if (!sync() || !readResult(result)) return {};
        if (!result.tracked) return result;
        if (!crop(out_targets_)) return {};

        status = vpiSubmitDCFTrackerUpdateBatch(stream_,
                                                VPI_BACKEND_CUDA,
                                                dcf_,
                                                nullptr,
                                                0,
                                                nullptr,
                                                nullptr,
                                                target_patch_,
                                                out_targets_,
                                                &params_);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to update DCF model", status);
            return {};
        }

        if (!sync()) return {};
        std::swap(in_targets_, out_targets_);
        return result;
    }

    bool DcfTracker::readResult(DcfTrackerResult& result) {
        VPIArrayData targets{};
        VPIStatus status = vpiArrayLockData(out_targets_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &targets);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to read DCF target", status);
            return false;
        }

        const auto* target = static_cast<const VPIDCFTrackedBoundingBox*>(targets.buffer.aos.data);
        const int32_t count = *targets.buffer.aos.sizePointer;

        if (count > 0) {
            result.box = cv::Rect2f{target[0].bbox.left,
                                    target[0].bbox.top,
                                    target[0].bbox.width,
                                    target[0].bbox.height};

            result.tracked = target[0].state != VPI_TRACKING_STATE_LOST;
        }

        status = vpiArrayUnlock(out_targets_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to unlock DCF target", status);
            return false;
        }

        VPIArrayData responses{};
        status = vpiArrayLockData(max_responses_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &responses);

        if (status != VPI_SUCCESS) {
            logVpiError("Failed to read DCF response", status);
            return false;
        }

        const int32_t response_count = *responses.buffer.aos.sizePointer;

        if (response_count > 0) {
            const auto* values = static_cast<const float*>(responses.buffer.aos.data);
            result.response = values[0];
        }

        status = vpiArrayUnlock(max_responses_);
        if (status != VPI_SUCCESS) {
            logVpiError("Failed to unlock DCF response", status);
            return false;
        }

        return true;
    }

    void DcfTracker::reset() noexcept {
        initialized_ = false;
        input_wrapper_.release();

        if (stream_ != nullptr) vpiStreamSync(stream_);
        if (max_responses_ != nullptr) {
            vpiArrayDestroy(max_responses_);
            max_responses_ = nullptr;
        }

        if (out_targets_ != nullptr) {
            vpiArrayDestroy(out_targets_);
            out_targets_ = nullptr;
        }

        if (in_targets_ != nullptr) {
            vpiArrayDestroy(in_targets_);
            in_targets_ = nullptr;
        }

        if (target_patch_ != nullptr) {
            vpiImageDestroy(target_patch_);
            target_patch_ = nullptr;
        }

        if (rgba_frame_ != nullptr) {
            vpiImageDestroy(rgba_frame_);
            rgba_frame_ = nullptr;
        }

        if (dcf_ != nullptr) {
            vpiPayloadDestroy(dcf_);
            dcf_ = nullptr;
        }

        if (crop_scaler_ != nullptr) {
            vpiPayloadDestroy(crop_scaler_);
            crop_scaler_ = nullptr;
        }

        if (stream_ != nullptr) {
            vpiStreamDestroy(stream_);
            stream_ = nullptr;
        }

        width_ = 0;
        height_ = 0;
        patch_size_ = 0;
    }

}
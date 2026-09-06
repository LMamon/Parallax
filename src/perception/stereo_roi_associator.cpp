#include <parallax/perception/stereo_roi_associator.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace parallax::perception {

    bool RectifiedCameraModel::valid() const noexcept {
        return std::isfinite(fx_px) &&
               std::isfinite(fy_px) &&
               std::isfinite(cx_px) &&
               std::isfinite(cy_px) &&
               fx_px > 0.0F &&
               fy_px > 0.0F && !coordinate_frame.empty();
    }

    StereoRoiAssociator::StereoRoiAssociator(const stereo::StereoCalibration& calibration, std::string coordinate_frame)
                                                : calibration_(calibration), coordinate_frame_(std::move(coordinate_frame)) {}

    bool StereoRoiAssociator::initialize() {
        if (!calibration_.loaded()) return false;

        const auto& metadata = calibration_.metadata();
        const auto& p1 = calibration_.P1();

        /*
         * Object3D is back-projected from RectifiedLeft depth coordinates.
         * P1 therefore owns the correct camera model here. Do not substitute
         * distorted-camera or virtual calibration values.
         */
        RectifiedCameraModel camera_model{};
        camera_model.fx_px = static_cast<float>(p1[0]);
        camera_model.fy_px = static_cast<float>(p1[5]);
        camera_model.cx_px = static_cast<float>(p1[2]);
        camera_model.cy_px = static_cast<float>(p1[6]);
        camera_model.coordinate_frame = coordinate_frame_;

        return initialize(metadata.image_width,
                         metadata.image_height,
                         calibration_.leftMapX(),
                         calibration_.leftMapY(),
                         std::move(camera_model));
    }

    bool StereoRoiAssociator::initialize(std::uint32_t width,
                                         std::uint32_t height,
                                         const std::vector<float>& rectified_to_rgb_x,
                                         const std::vector<float>& rectified_to_rgb_y,
                                         RectifiedCameraModel camera_model) {

        initialized_ = false;
        if (width == 0 || height == 0 || !camera_model.valid()) {
            return false;
        }

        if (!mapper_.initialize(width, height, rectified_to_rgb_x, rectified_to_rgb_y)) {
            return false;
        }

        /*
        * Mask-supported Object3D refinement operates in rectified depth space
        * while segmentation remains in the source RGB image space. Preserve the
        * calibration's rectified->source maps on device so mask/depth association
        * does not require a per-frame host mapping pass.
        */
        if (!rectified_to_rgb_x_device_.allocate(width, height, 1, sizeof(float)) ||
            !rectified_to_rgb_y_device_.allocate( width, height, 1, sizeof(float))) {
            return false;
        }
        if (!surface_samples_device_.allocate(static_cast<std::uint32_t>(MaxSurfaceSamples), 1, 1, sizeof(cuda::MaskedDepthPoint)) ||
            !surface_sample_count_device_.allocate(1, 1, 1, sizeof(std::uint32_t))) {

            return false;
        }
        
        cudaStream_t upload_stream = nullptr;
        if (cudaStreamCreate(&upload_stream) != cudaSuccess) return false;

        const std::size_t host_pitch = static_cast<std::size_t>(width) * sizeof(float);
        const bool uploaded = rectified_to_rgb_x_device_.uploadAsync(rectified_to_rgb_x.data(),
                                                                    host_pitch,
                                                                    upload_stream) &&
                            rectified_to_rgb_y_device_.uploadAsync(rectified_to_rgb_y.data(),
                                                                    host_pitch,
                                                                    upload_stream);

        const bool synchronized = uploaded && cudaStreamSynchronize(upload_stream) == cudaSuccess;

        cudaStreamDestroy(upload_stream);
        if (!synchronized) return false;

        /*
         * Scratch storage is allocated once for the maximum bounded batch.
         * The association path never grows device memory because detector
         * output increased unexpectedly.
         */
        if (!requests_device_.allocate(static_cast<std::uint32_t>(MaxObjects), 1, 1, sizeof(cuda::DepthRoiRequest))) {
            return false;
        }

        if (!results_device_.allocate(static_cast<std::uint32_t>(MaxObjects), 1, 1, sizeof(cuda::DepthRoiResult))) {
            requests_device_.release();
            return false;
        }

        image_width_ = width;
        image_height_ = height;
        camera_model_ = std::move(camera_model);

        initialized_ = true;
        return true;
    }

    bool StereoRoiAssociator::associate(const DetectionSet& detections,
                                        const core::ProductMetadata& semantic_metadata,
                                        const core::Product<isp::DepthFrame>& depth,
                                        core::ExecutionContext& context,
                                        Object3DSet& output) {

        output = {};

        if (!initialized_ ||
            !detections.valid() ||
            !semantic_metadata.valid ||
            !semantic_metadata.observation.valid() ||
            !depth.valid() || !depth.payload) {
            return false;
        }

        /*
         * This class performs metric association, not temporal compatibility
         * selection. The caller must supply the Depth generation selected by
         * the Phase 16 compatibility policy.
         */
        if (!depth.metadata.valid || !depth.metadata.observation.valid()) return false;

        if (depth.payload->width != image_width_ ||
            depth.payload->height != image_height_ ||
            !depth.payload->depth.isAllocated()) {
            return false;
        }

        /*
         * MaxObjects is a real bounded-work contract. Silently truncating a
         * larger detector result would make output completeness ambiguous.
         */
        if (detections.size() > MaxObjects) return false;

        output.query = detections.query;
        output.query_revision = detections.query_revision;
        if (detections.empty()) return true;

        std::array<std::size_t, MaxObjects> detection_indices{};
        std::array<cv::Point2f, MaxObjects> rectified_centers{};

        std::uint32_t request_count = 0;

        /*
         * NanoOWL boxes remain expressed in their original semantic image
         * space. Only the sampling center is transformed into RectifiedLeft.
         * Object3D keeps the original image box for provenance/visualization.
         */
        for (std::size_t i = 0; i < detections.size(); ++i) {
            const auto& box = detections.boxes[i];
            const float score = detections.scores[i];

            if (!std::isfinite(box.x) ||
                !std::isfinite(box.y) ||
                !std::isfinite(box.width) ||
                !std::isfinite(box.height) ||
                box.width <= 0.0F ||
                box.height <= 0.0F || !std::isfinite(score)) {
                continue;
            }

            const cv::Point2f semantic_center{box.x + box.width * 0.5F, box.y + box.height * 0.5F};
            cv::Point2f rectified_center{};

            /*
             * A detection near calibration/crop boundaries can be semantically
             * valid but have no supported rectified depth coordinate. Skip that
             * object rather than inventing a metric position.
             */
            if (!mapper_.mapPoint(semantic_center, detections.image_space, ImageSpace::RectifiedLeft, rectified_center)) {
                continue;
            }

            requests_host_[request_count] = {static_cast<std::int32_t>(std::lround(rectified_center.x)),
                                             static_cast<std::int32_t>(std::lround(rectified_center.y))};

            detection_indices[request_count] = i;
            rectified_centers[request_count] = rectified_center;

            ++request_count;
        }

        if (request_count == 0) return true;

        auto& lane = context.stereoLane();
        const cudaStream_t stream = lane.cudaHandle();

        if (stream == nullptr) return false;

        // Depth may still be in flight when the product is visible in the
        // store. Add a generation-specific accelerator dependency instead of
        // globally synchronizing stereo or the device.
        if (!context.waitFor(depth.completion, lane)) return false;


        // CudaBuffer copies the fixed scratch row. Even though only
        // request_count entries are consumed by the kernel, keeping a fixed
        // allocation/copy shape avoids per-frame allocation and remains tiny.        
        const std::size_t request_host_pitch = MaxObjects * sizeof(cuda::DepthRoiRequest);
        if (!requests_device_.uploadAsync(requests_host_.data(), request_host_pitch, stream)) {
            return false;
        }

        if (!cuda::reduceDepthRois(depth.payload->depth,
                                  requests_device_,
                                  results_device_,
                                  request_count,
                                  RoiRadius,
                                  stream)) {

            return false;
        }

        const std::size_t result_host_pitch = MaxObjects * sizeof(cuda::DepthRoiResult);
        if (!results_device_.downloadAsync(results_host_.data(), result_host_pitch, stream)) {
            return false;
        }

        // CPU Object3D metadata needs the compact ROI results. Synchronize only
        // this submitted CUDA chain, after the full depth image has already
        // been reduced to at most MaxObjects tiny structs.
        auto completion = context.recordCudaCompletion(stream);
        if (!completion.valid() || !context.waitForHost(completion)) return false;

        output.objects.reserve(request_count);

        const auto association_timestamp = std::chrono::steady_clock::now();

        const auto source_delta = depth.metadata.timestamp >= semantic_metadata.timestamp
                                  ? depth.metadata.timestamp - semantic_metadata.timestamp
                                  : semantic_metadata.timestamp - depth.metadata.timestamp;

        for (std::uint32_t request = 0; request < request_count; ++request) {
            const auto& roi = results_host_[request];

            // Sparse stereo may leave the center invalid while surrounding
            // pixels remain useful. Require a minimum support count instead of
            // requiring one special pixel to be valid.
            if (roi.valid_samples < MinValidSamples ||
                roi.sampled_pixels == 0 ||
                !std::isfinite(roi.depth_m) || roi.depth_m <= 0.0F) {

                continue;
            }

            std::array<float, 3> xyz{};
            if (!backProject(rectified_centers[request], roi.depth_m, xyz)) continue;

            const auto& request_center = requests_host_[request];

            const int x0 = std::max(0, request_center.center_x - static_cast<int>(RoiRadius));
            const int y0 = std::max(0, request_center.center_y - static_cast<int>(RoiRadius));

            const int x1 = std::min(static_cast<int>(image_width_) - 1, request_center.center_x + static_cast<int>(RoiRadius));
            const int y1 = std::min(static_cast<int>(image_height_) - 1, request_center.center_y + static_cast<int>(RoiRadius));

            const std::size_t detection_index = detection_indices[request];

            // A 2D detection box plus representative stereo depth supports a planar
            // image-space rectangle in 3D. It does not establish physical object extent.
            std::array<std::array<float, 3>, 4> image_supported_corners_m{};

            const auto& source_box = detections.boxes[detection_index];

            const cv::Point2f source_corners[4] = {
                {source_box.x, source_box.y},
                {source_box.x + source_box.width, source_box.y},
                {source_box.x + source_box.width, source_box.y + source_box.height},
                {source_box.x, source_box.y + source_box.height}
            };

            bool rectangle_valid = true;

            for (std::size_t corner = 0; corner < 4; ++corner) {
                cv::Point2f rectified_corner{};

                if (!mapper_.mapPoint(source_corners[corner],
                                    detections.image_space,
                                    ImageSpace::RectifiedLeft,
                                    rectified_corner) ||
                            !backProject(rectified_corner,
                                        roi.depth_m,
                                        image_supported_corners_m[corner])) {

                    rectangle_valid = false;
                    break;
                }
            }

            Object3D object{};
            object.label = detections.query;
            object.query_revision = detections.query_revision;
            object.semantic_confidence = detections.scores[detection_index];
            object.semantic_index = static_cast<std::uint32_t>(detection_index);
            object.image_box = detections.boxes[detection_index];
            object.image_space = detections.image_space;

            object.position_m = xyz;
            object.image_supported_corners_m = image_supported_corners_m;
            object.depth_m = roi.depth_m;
            object.coordinate_frame = camera_model_.coordinate_frame;


            object.geometry = rectangle_valid ? Object3DGeometry::ImageSupportedGeometry : Object3DGeometry::Point;
            object.method = Object3DMethod::StereoRoi;

            object.semantic_observation = semantic_metadata.observation;
            object.metric_observation = depth.metadata.observation;
            object.association_timestamp = association_timestamp;
            object.source_time_delta = source_delta;

            object.depth_roi = {static_cast<float>(x0),
                                static_cast<float>(y0),
                                static_cast<float>(x1 - x0 + 1),
                                static_cast<float>(y1 - y0 + 1)};

            object.depth_image_space = ImageSpace::RectifiedLeft;

            // This is support density, not semantic confidence. Keep those
            // quality signals separate so downstream policy can reason about
            // detector confidence and stereo support independently.
            object.support_quality = static_cast<float>(roi.valid_samples) / static_cast<float>(roi.sampled_pixels);

            output.objects.push_back(std::move(object));
        }
        return true;
    }

    bool StereoRoiAssociator::refineWithMask(const SegmentationMask& mask,
                                             const core::ProductMetadata& mask_metadata,
                                             const core::Product<isp::DepthFrame>& depth,
                                             core::ExecutionContext& context,
                                             Object3D& object) {

        if (!initialized_ ||
            !object.valid() ||
            !mask.valid() ||
            !mask_metadata.valid ||
            !depth.valid() ||
            !depth.payload ||
            mask.representation != MaskRepresentation::CudaDevice ||
            mask.layout != MaskLayout::RowMajor ||
            mask.image_space != ImageSpace::RgbLeft ||
            mask.source_observation != object.semantic_observation ||
            mask.query_revision != object.query_revision) {

            return false;
        }

        /*
        * Segmentation and depth are independently asynchronous CUDA products.
        * Chain both generations onto the stereo lane before reading either.
        */
        auto& lane = context.stereoLane();
        const cudaStream_t stream = lane.cudaHandle();

        if (stream == nullptr || !context.waitFor(depth.completion, lane)) {
            return false;
        }

        /*
        * The producer supplies mask metadata separately because the mask payload
        * intentionally stores only source/query identity, not its completion.
        */
        (void)mask_metadata;
        const auto* mask_device = static_cast<const std::uint8_t*>(mask.storage.get());

        if (!cuda::sampleMaskedDepth(mask_device,
                                     mask.pitch_bytes,
                                     mask.width,
                                     mask.height,
                                     depth.payload->depth,
                                     rectified_to_rgb_x_device_,
                                     rectified_to_rgb_y_device_,
                                     camera_model_.fx_px,
                                     camera_model_.fy_px,
                                     camera_model_.cx_px,
                                     camera_model_.cy_px,
                                     SurfaceSampleStride,
                                     static_cast<std::uint32_t>(MaxSurfaceSamples),
                                     surface_samples_device_,
                                     surface_sample_count_device_,
                                     stream)) {

            return false;
        }

        if (!surface_samples_device_.downloadAsync(surface_samples_host_.data(),
                                                    MaxSurfaceSamples * sizeof(cuda::MaskedDepthPoint),
                                                    stream) ||
            !surface_sample_count_device_.downloadAsync(&surface_sample_count_host_,
                                                        sizeof(std::uint32_t), 
                                                        stream)) {

            return false;
        }

        auto completion = context.recordCudaCompletion(stream);
        if (!completion.valid() || !context.waitForHost(completion)) return false;

        const std::size_t count = std::min<std::size_t>(surface_sample_count_host_, MaxSurfaceSamples);
        if (count < MinSurfaceSamples) return false;

        object.surface_points_m.clear();
        object.surface_points_m.reserve(count);

        std::vector<float> depths;
        depths.reserve(count);

        std::array<float, 3> centroid{};

        for (std::size_t i = 0; i < count; ++i) {
            const auto& sample = surface_samples_host_[i];

            if (!std::isfinite(sample.x) || !std::isfinite(sample.y) || !std::isfinite(sample.z) || sample.z <= 0.0F) {
                continue;
            }

            object.surface_points_m.push_back({sample.x, sample.y, sample.z});

            centroid[0] += sample.x;
            centroid[1] += sample.y;
            centroid[2] += sample.z;

            depths.push_back(sample.z);
        }

        if (object.surface_points_m.size() < MinSurfaceSamples) {
            object.surface_points_m.clear();
            return false;
        }

        const float reciprocal = 1.0F / static_cast<float>(object.surface_points_m.size());

        centroid[0] *= reciprocal;
        centroid[1] *= reciprocal;
        centroid[2] *= reciprocal;

        std::sort(depths.begin(), depths.end());

        const std::size_t middle = depths.size() / 2;
        const float median_depth = (depths.size() & 1U) ? depths[middle] : 0.5F * (depths[middle - 1] + depths[middle]);

        /*
        * The mask-supported sample refines the representative object position
        * without claiming a watertight surface or physical object dimensions.
        */
        object.position_m = centroid;
        object.depth_m = median_depth;
        object.geometry = Object3DGeometry::Surface;
        object.method = Object3DMethod::StereoMask;

        object.support_quality = static_cast<float>(object.surface_points_m.size()) / static_cast<float>(MaxSurfaceSamples);

        return true;
    }

    bool StereoRoiAssociator::backProject(const cv::Point2f& rectified_point, float depth_m, std::array<float, 3>& xyz) const noexcept {
        if (!camera_model_.valid() ||
            !std::isfinite(rectified_point.x) ||
            !std::isfinite(rectified_point.y) ||
            !std::isfinite(depth_m) || depth_m <= 0.0F) {

            return false;
        }

        // P1 describes the rectified left pinhole camera. Depth is Z in that
        // same camera frame, so no additional calibration transform is needed.
        xyz[0] = (rectified_point.x - camera_model_.cx_px) * depth_m / camera_model_.fx_px;
        xyz[1] = (rectified_point.y - camera_model_.cy_px) * depth_m / camera_model_.fy_px;
        xyz[2] = depth_m;

        return std::isfinite(xyz[0]) && std::isfinite(xyz[1]) && std::isfinite(xyz[2]);
    }
}
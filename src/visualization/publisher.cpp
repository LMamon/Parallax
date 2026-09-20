#include <parallax/visualization/publisher.hpp>
#include <parallax/pose/charuco_pose.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/tracking/track.hpp>


#include <algorithm>
#include <cstring>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
#include <chrono>
#include <array>
#include <iomanip>
#include <cmath>
#include <utility>

namespace parallax::visualization {
    namespace {
        bool checkFoxglove(const foxglove::FoxgloveError& error, const char* message) {
            if (error != foxglove::FoxgloveError::Ok) {
                std::cerr << message << ": " << foxglove::strerror(error) << '\n';
                return false;
            }
            return true;
        }

        foxglove::messages::Timestamp nowTimestamp() {
            const auto now = std::chrono::system_clock::now();
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

            foxglove::messages::Timestamp timestamp;
            timestamp.sec = static_cast<std::int32_t>(ns / 1'000'000'000LL);
            timestamp.nsec = static_cast<std::uint32_t>(ns % 1'000'000'000LL);

            return timestamp;
        }

        foxglove::messages::Timestamp sourceTimestamp(const parallax::core::ProductMetadata& metadata) {
            if (!metadata.wall_timestamp_valid) return nowTimestamp();
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(metadata.wall_timestamp.time_since_epoch()).count();
            if (ns < 0) return nowTimestamp();
            foxglove::messages::Timestamp timestamp;
            timestamp.sec = static_cast<std::int32_t>(ns / 1'000'000'000LL);
            timestamp.nsec = static_cast<std::uint32_t>(ns % 1'000'000'000LL);
            return timestamp;
        }
        
        foxglove::messages::FrameTransform makeFrameTransform(const parallax::core::RigidTransformConfig& config) {
                foxglove::messages::FrameTransform message;
                message.parent_frame_id = config.parent_frame;
                message.child_frame_id = config.child_frame;

                foxglove::messages::Vector3 translation;
                translation.x = config.translation_m[0];
                translation.y = config.translation_m[1];
                translation.z = config.translation_m[2];
                message.translation = translation;
                
                foxglove::messages::Quaternion rotation;
                rotation.x = config.rotation_xyzw[0];
                rotation.y = config.rotation_xyzw[1];
                rotation.z = config.rotation_xyzw[2];
                rotation.w = config.rotation_xyzw[3];
                
                message.rotation = rotation;

                return message;
        }
        
        const char* sourceIdName(parallax::core::SourceId source) noexcept {
            switch (source) {
                case parallax::core::SourceId::StereoCamera:
                    return "stereo_camera";

                case parallax::core::SourceId::Rplidar:
                    return "rplidar";

                case parallax::core::SourceId::Unknown:
                default:
                    return "unknown";
            }
        }

        const char* objectMetricSourceName(parallax::perception::Object3DMethod method) noexcept {
            using Method = parallax::perception::Object3DMethod;

            switch (method) {
                case Method::LidarAssociation:
                    return "lidar";
                case Method::StereoMask:
                    return "stereo_mask";
                case Method::StereoRoi:
                    return "stereo";
                case Method::StereoLidarRefined:
                    return "stereo_lidar";
                case Method::Unknown:
                default:
                    return "unknown";
            }
        }

        float objectDisplayDistance(const parallax::perception::Object3D& object) noexcept {
            if (object.method == parallax::perception::Object3DMethod::LidarAssociation &&
                std::isfinite(object.range_m) &&
                object.range_m > 0.0F) {
                return object.range_m;
            }
            return object.depth_m;
        }
    
        std::string formatDepth(float depth_m) {
            if (!std::isfinite(depth_m) || depth_m <= 0.0F) return {};

            constexpr float MetersToFeet = 3.280839895F;
            constexpr float FeetToInches = 12.0F;
            const float feet = depth_m * MetersToFeet;

            std::ostringstream stream;
            stream << std::fixed;

            if (feet < 1.0F) {
                stream << std::setprecision(1) << feet * FeetToInches << " in";
            } else if (feet < 3.0F) {
                stream << std::setprecision(1) << feet << " ft";
            } else {
                stream << std::setprecision(2) << depth_m << " m";
            }

            return stream.str();
        }

    }

    Publisher::~Publisher() { shutdown(); }

    bool Publisher::initialize(FoxgloveServer& foxglove, 
                               std::uint32_t width, 
                               std::uint32_t height,
                               std::uint32_t fps,
                               std::string coordinate_frame,
                               const parallax::stereo::StereoCalibration& calibration,
                               std::uint32_t preview_fps,
                               int jpeg_quality) {

        if (initialized_) return true;
        if (width == 0 || height == 0 || fps == 0 || preview_fps == 0 ||
            jpeg_quality < 1 || jpeg_quality > 100 || coordinate_frame.empty() ||
            !calibration.loaded()) {
            std::cerr << "Invalid visualization dimensions/FPS\n";
            return false;
        }

        if (!foxglove.initialized()) {
            std::cerr << "Foxglove server must be initialized before Publisher\n";
            return false;
        }

        foxglove_ = &foxglove;
        width_ = width;
        height_ = height;
        fps_ = fps;
        preview_fps_ = preview_fps;
        jpeg_quality_ = jpeg_quality;
        coordinate_frame_ = std::move(coordinate_frame);

        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            std::cerr << "Failed to create visualization CUDA stream\n";
            shutdown();
            return false;
        }

        const std::size_t pixels = static_cast<std::size_t>(width_) * height_;
        const std::size_t rgb_bytes = pixels * 3 * sizeof(std::uint8_t);

        const std::size_t disparity_bytes = pixels * sizeof(std::int16_t);
        depth_preview_width_ = (width_ + DepthPreviewStride - 1U) / DepthPreviewStride;
        depth_preview_height_ = (height_ + DepthPreviewStride - 1U) / DepthPreviewStride;

        // Nearest preview samples are (0, stride, 2*stride, ...), so scale P1 directly.
        depth_preview_projection_ = calibration.P1();
        const double preview_scale = 1.0 / static_cast<double>(DepthPreviewStride);
        depth_preview_projection_[0] *= preview_scale;
        depth_preview_projection_[2] *= preview_scale;
        depth_preview_projection_[3] *= preview_scale;
        depth_preview_projection_[5] *= preview_scale;
        depth_preview_projection_[6] *= preview_scale;
        depth_preview_projection_[7] *= preview_scale;

        const std::size_t depth_preview_bytes = static_cast<std::size_t>(depth_preview_width_) *
                                                depth_preview_height_ * sizeof(float);
        const std::size_t mask_bytes =static_cast<std::size_t>(width_) * height_ * sizeof(std::uint8_t);


        if (cudaMallocHost(reinterpret_cast<void**>(&host_segmentation_mask_), mask_bytes) != cudaSuccess) {
            std::cerr << "Failed to allocate segmentation mask staging buffer\n";
            shutdown();
            return false;
        }

        if (cudaMallocHost(reinterpret_cast<void**>(&host_rgb_), rgb_bytes) != cudaSuccess) {
            std::cerr << "Failed to allocate RGB staging buffer\n";
            shutdown();
            return false;
        }

        if (cudaMallocHost(reinterpret_cast<void**>(&host_disparity_), disparity_bytes) != cudaSuccess) {
            std::cerr << "Failed to allocate disparity staging buffer\n";
            shutdown();
            return false;
        }

        if (!depth_preview_.allocate(depth_preview_width_, depth_preview_height_, 1, sizeof(float))) {
            std::cerr << "Failed to allocate depth preview buffer\n";
            shutdown();
            return false;
        }

        if (cudaMallocHost(reinterpret_cast<void**>(&host_depth_), depth_preview_bytes) != cudaSuccess) {
            std::cerr << "Failed to allocate depth preview staging buffer\n";
            shutdown();
            return false;
        }

        disparity_float_.resize(pixels);
        bgr_storage_.resize(rgb_bytes);
        jpeg_bytes_.reserve(rgb_bytes / 4U);

        initialized_ = true;
        return true;
    }

    bool Publisher::publishLeftCalibration(const parallax::stereo::StereoCalibration& calibration) {
        if (!initialized_ || foxglove_ == nullptr || !calibration.loaded()) {
            return false;
        }

        const auto& metadata = calibration.metadata();
        const auto& p1 = calibration.P1();

        foxglove::messages::CameraCalibration message;

        message.frame_id = coordinate_frame_;
        message.width = metadata.image_width;
        message.height = metadata.image_height;

        // The published image is already rectified.
        message.distortion_model = "plumb_bob";
        message.d = {0.0, 0.0, 0.0, 0.0, 0.0};

        // Intrinsics of the rectified virtual camera.
        message.k = {p1[0], p1[1], p1[2],
                    p1[4], p1[5], p1[6],
                    p1[8], p1[9], p1[10]};

        // Image has already undergone the stereo rectification transform.
        message.r = {1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0};

        message.p = p1;

        return checkFoxglove(foxglove_->leftCalibrationChannel().log(message), "Failed to publish /camera/left/calibration");
    }

    bool Publisher::publishDepthCalibration(const parallax::stereo::StereoCalibration& calibration) {
        if (!initialized_ || foxglove_ == nullptr || !calibration.loaded()) {
            return false;
        }

        const auto& p1 = calibration.P1();
        const double scale = 1.0 / static_cast<double>(DepthPreviewStride);

        foxglove::messages::CameraCalibration message;
        message.frame_id = coordinate_frame_;
        message.width = depth_preview_width_;
        message.height = depth_preview_height_;
        message.distortion_model = "plumb_bob";
        message.d = {0.0, 0.0, 0.0, 0.0, 0.0};
        message.k = {p1[0] * scale, p1[1],         p1[2] * scale,
                     p1[4],         p1[5] * scale, p1[6] * scale,
                     p1[8],         p1[9],         p1[10]};
        message.r = {1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0};
        message.p = {p1[0] * scale, p1[1],         p1[2] * scale, p1[3] * scale,
                     p1[4],         p1[5] * scale, p1[6] * scale, p1[7] * scale,
                     p1[8],         p1[9],         p1[10],        p1[11]};

        return checkFoxglove(foxglove_->depthCalibrationChannel().log(message),
                             "Failed to publish /stereo/depth/calibration");
    }

    bool Publisher::publishStaticTransforms(const parallax::core::SensorExtrinsics& extrinsics) {
        if (!initialized_ || foxglove_ == nullptr) return false;

        const auto left_camera = makeFrameTransform(extrinsics.left_camera);
        if (!checkFoxglove(foxglove_->transformChannel().log(left_camera),
                "Failed to publish stereo_body -> camera_left_optical transform")) {

            return false;
        }

        const auto lidar = makeFrameTransform(extrinsics.lidar);

        if (!checkFoxglove(foxglove_->transformChannel().log(lidar),
                           "Failed to publish stereo_body -> lidar transform")) {

            return false;
        }
        return true;
    }

    bool Publisher::publishAvailable(const parallax::core::ProductStore& store, const HostWait& wait_for_host) {
        if (!initialized_ || foxglove_ == nullptr || !wait_for_host) {
            return false;
        }

        /**
         * RGB is selected independently from the overlay.
         *
         * Visualization must never manufacture marker-pose demand. If the newest
         * marker result belongs to this exact camera observation, it may decorate
         * the image. Otherwise the image is published without an overlay.
         */
        if (foxglove_->leftImageChannel().hasSinks()) {
            const auto rgb = store.latest<parallax::isp::RectifiedStereoFrame>(
                                                parallax::core::ProductId::RectifiedRgb);

            if (rgb && rgb->valid()) {
                const auto now = std::chrono::steady_clock::now();
                const auto interval = std::chrono::duration<double>(1.0 / static_cast<double>(preview_fps_));
                const bool new_observation = !has_published_left_image_ || rgb->metadata.observation != last_left_image_observation_;
                const bool due = !has_published_left_image_ || now - last_left_image_publish_ >= interval;
                if (new_observation && due) {
                    const parallax::pose::CharucoPoseResult* overlay = nullptr;
                    const auto marker = store.latest<parallax::pose::CharucoPoseResult>(parallax::core::ProductId::MarkerDepth);
                    if (marker && marker->valid() && parallax::core::same_source_observation(*marker, rgb->metadata.observation)) overlay = marker->payload.get();
                    if (!rgb->completion.valid()) return false;
                    if (rgb->completion.requires_wait() && !wait_for_host(rgb->completion)) return false;
                    if (!publishLeftImage(*rgb, overlay)) return false;
                    last_left_image_observation_ = rgb->metadata.observation;
                    last_left_image_publish_ = now;
                    has_published_left_image_ = true;
                }
            }
        }

        /**
         * Disparity and confidence are two views of the same StereoMatchFrame
         * product. Only observe/download it if at least one corresponding
         * Foxglove channel actually has a sink.
         */
        const bool disparity_requested = foxglove_->disparityChannel().hasSinks();
        if (disparity_requested) {
            const auto stereo = store.latest<parallax::isp::StereoMatchFrame>(parallax::core::ProductId::Disparity);

            if (stereo && stereo->valid() && (!has_published_disparity_ || stereo->metadata.observation != last_disparity_observation_)) {
                if (!stereo->completion.valid()) return false;
                if (stereo->completion.requires_wait() && !wait_for_host(stereo->completion)) {
                    return false;
                }

                if (disparity_requested && !publishDisparity(*stereo->payload)) return false;
                last_disparity_observation_ = stereo->metadata.observation;
                has_published_disparity_ = true;
            }
        }

        const bool depth_image_requested = foxglove_->depthChannel().hasSinks();
        const bool depth_scene_requested = foxglove_->depthSceneChannel().hasSinks();

        if (depth_image_requested || depth_scene_requested) {
            const auto depth = store.latest<parallax::isp::DepthFrame>(parallax::core::ProductId::Depth);

            if (depth && depth->valid() && (!has_published_depth_ || depth->metadata.observation != last_depth_observation_)) {
                const auto now = std::chrono::steady_clock::now();
                const auto interval = std::chrono::duration<double>(1.0 / static_cast<double>(DepthPreviewFps));
                const bool due = !has_published_depth_ || now - last_depth_publish_ >= interval;

                if (due) {
                    if (!depth->completion.valid()) return false;
                    if (depth->completion.requires_wait() && !wait_for_host(depth->completion)) return false;
                    if (!publishDepth(*depth, depth_image_requested, depth_scene_requested)) return false;
                    last_depth_observation_ = depth->metadata.observation;
                    last_depth_publish_ = now;
                    has_published_depth_ = true;
                }
            }
        }

        if (foxglove_->lidarScanChannel().hasSinks()) {
            const auto lidar = store.latest<parallax::lidar::LidarScan>(parallax::core::ProductId::LidarScan);

            if (lidar && lidar->valid() && (!has_published_lidar_ || lidar->metadata.observation != last_lidar_observation_)) {
                if (!publishLidarScan(*lidar->payload)) return false;
                last_lidar_observation_ = lidar->metadata.observation;
                has_published_lidar_ = true;
            }
        }

        if (foxglove_->detectionChannel().hasSinks()) {
            const auto detections = store.latest<parallax::perception::DetectionSet>(parallax::core::ProductId::Detection);
            
            if (detections && detections->valid()) {
                const bool new_observation = !has_published_detection_ ||
                                              detections->metadata.observation != last_detection_observation_ ||
                                              detections->payload->query_revision != last_detection_query_revision_;
                if (new_observation) {
                    if (!publishDetections(*detections)) return false;

                    last_detection_observation_ = detections->metadata.observation;
                    last_detection_query_revision_ = detections->payload->query_revision;
                    has_published_detection_ = true;
                }
            }
        }

        if (foxglove_->detectionAnnotationsChannel().hasSinks()) {
            const auto detections = store.latest<parallax::perception::DetectionSet>(parallax::core::ProductId::Detection);

            if (detections && detections->valid()) {
                const bool new_annotation = !has_published_detection_annotation_ ||
                                            detections->metadata.observation != last_detection_annotation_observation_ ||
                                            detections->payload->query_revision != last_detection_annotation_query_revision_;

                if (new_annotation) {
                    if (!publishDetectionAnnotations(*detections)) return false;
                    
                    last_detection_annotation_observation_ = detections->metadata.observation;
                    last_detection_annotation_query_revision_ = detections->payload->query_revision;
                    has_published_detection_annotation_ = true;
                }
            }
        }

        if (foxglove_->trackAnnotationsChannel().hasSinks()) {
            const auto track = store.latest<parallax::tracking::Track2D>(parallax::core::ProductId::Track2D);

            if (track && track->valid() && track->payload->valid()) {

                const bool new_annotation = !has_published_track_annotation_ ||
                                            track->metadata.observation != last_track_annotation_observation_ ||
                                            track->payload->target_revision != last_track_annotation_revision_ ||
                                            track->payload->lifecycle != last_track_annotation_lifecycle_;

                if (new_annotation) {
                    if (!publishTrackAnnotations(*track)) return false;

                    last_track_annotation_observation_ = track->metadata.observation;
                    last_track_annotation_revision_ = track->payload->target_revision;
                    last_track_annotation_lifecycle_ = track->payload->lifecycle;
                    has_published_track_annotation_ = true;
                }
            }
        }

        if (foxglove_->segmentationMaskChannel().hasSinks()) {
            const auto segmentation = store.latest<parallax::perception::SegmentationMask>(
                                                   parallax::core::ProductId::Segmentation);

            if (segmentation && segmentation->valid() && segmentation->payload->valid()) {

                const bool new_segmentation = !has_published_segmentation_ ||
                                               segmentation->metadata.observation !=
                                               last_segmentation_observation_ ||
                                               segmentation->payload->query_revision !=
                                               last_segmentation_query_revision_;

                if (new_segmentation) {
                    if (!segmentation->completion.valid()) return false;
                    if (segmentation->completion.requires_wait() && !wait_for_host(segmentation->completion)) {
                        return false;
                    }

                    if (!publishSegmentationMask(*segmentation)) return false;

                    last_segmentation_observation_ = segmentation->metadata.observation;
                    last_segmentation_query_revision_ = segmentation->payload->query_revision;
                    has_published_segmentation_ = true;
                }
            }
        }

        if (foxglove_->objectDepthAnnotationsChannel().hasSinks()) {
            const auto objects = store.latest<parallax::perception::Object3DSet>(parallax::core::ProductId::Object3D);

            if (objects && objects->valid() && objects->payload->valid()) {
                const bool new_annotation = !has_published_object_depth_annotation_ ||
                                            objects->metadata.observation != last_object_depth_annotation_observation_ ||
                                            objects->payload->query_revision != last_object_depth_annotation_revision_;

                if (new_annotation) {
                    if (!publishObjectDepthAnnotations(*objects)) {
                        return false;
                    }

                    last_object_depth_annotation_observation_ = objects->metadata.observation;
                    last_object_depth_annotation_revision_ = objects->payload->query_revision;
                    has_published_object_depth_annotation_ = true;
                }
            }
        }

        if (foxglove_->object3DSceneChannel().hasSinks()) {
            const auto objects = store.latest<parallax::perception::Object3DSet>(parallax::core::ProductId::Object3D);

            if (objects && objects->valid() && objects->payload->valid()) {
                const bool new_scene = !has_published_object_scene_ ||
                                        objects->metadata.observation != last_object_scene_observation_ ||
                                        objects->payload->query_revision != last_object_scene_revision_;

                if (new_scene) {
                    if (!publishObject3DScene(*objects)) {
                        return false;
                    }

                    last_object_scene_observation_ = objects->metadata.observation;
                    last_object_scene_revision_ = objects->payload->query_revision;
                    has_published_object_scene_ = true;
                }
            }
        }


        if (foxglove_->trackedObject3DSceneChannel().hasSinks()) {
            const auto tracked = store.latest<parallax::perception::Object3DSet>(
                                                parallax::core::ProductId::TrackedObject3D);

            if (tracked && tracked->valid() && tracked->payload &&
                tracked->payload->valid() && !tracked->payload->objects.empty()) {

                const auto& object = tracked->payload->objects.front();
                const bool new_scene = !has_published_tracked_object_scene_ ||
                                       tracked->metadata.observation != last_tracked_object_scene_observation_ ||
                                       tracked->payload->query_revision != last_tracked_object_scene_revision_ ||
                                       object.track_id != last_tracked_object_scene_track_id_ ||
                                       object.method != last_tracked_object_scene_method_;

                if (new_scene) {
                    if (!publishTrackedObject3DScene(*tracked)) return false;

                    last_tracked_object_scene_observation_ = tracked->metadata.observation;
                    last_tracked_object_scene_revision_ = tracked->payload->query_revision;
                    last_tracked_object_scene_track_id_ = object.track_id;
                    last_tracked_object_scene_method_ = object.method;
                    has_published_tracked_object_scene_ = true;
                }
            }
        }

        if (foxglove_->segmentationSceneChannel().hasSinks()) {
            const auto segmented = store.latest<parallax::perception::Object3DSet>(
                                                parallax::core::ProductId::TrackedObject3D);

            if (segmented && segmented->valid() && segmented->payload &&
                segmented->payload->valid() && !segmented->payload->objects.empty()) {

                const auto& object = segmented->payload->objects.front();

                /*
                 * Only the synchronized SAM+stereo correction belongs on this
                 * topic. Ordinary DCF rectangle support stays exclusively on
                 * /tracking/objects3d.
                 */
                if (object.method == parallax::perception::Object3DMethod::StereoMask) {
                    const bool new_scene = !has_published_segmentation_scene_ ||
                                           segmented->metadata.observation != last_segmentation_scene_observation_ ||
                                           segmented->payload->query_revision != last_segmentation_scene_revision_;

                    if (new_scene) {
                        if (!publishObject3DScene(*segmented->payload,
                                                  sourceTimestamp(segmented->metadata),
                                                  foxglove_->segmentationSceneChannel(),
                                                  "Failed to publish /perception/segmentation/scene")) {
                            return false;
                        }

                        last_segmentation_scene_observation_ = segmented->metadata.observation;
                        last_segmentation_scene_revision_ = segmented->payload->query_revision;
                        has_published_segmentation_scene_ = true;
                    }
                }
            }
        }

        if (foxglove_->localizedObject3DSceneChannel().hasSinks()) {
            const auto localized = store.latest<parallax::perception::LocalizedSpatialObservation>(
                                                parallax::core::ProductId::LocalizedSpatialObservation);

            if (localized && localized->valid() && localized->payload && localized->payload->valid()) {

                const bool new_scene = !has_published_localized_object_scene_ ||
                                        localized->metadata.observation != last_localized_object_scene_observation_ ||
                                        localized->payload->localization_observation != last_localized_object_scene_pose_observation_ ||
                                        localized->payload->objects.query_revision != last_localized_object_scene_revision_ ||
                                        localized->payload->localization_epoch != last_localized_object_scene_epoch_;

                if (new_scene) {
                    if (!publishLocalizedObject3DScene(*localized)) return false;

                    last_localized_object_scene_observation_ = localized->metadata.observation;
                    last_localized_object_scene_pose_observation_ = localized->payload->localization_observation;
                    last_localized_object_scene_revision_ = localized->payload->objects.query_revision;
                    last_localized_object_scene_epoch_ = localized->payload->localization_epoch;

                    has_published_localized_object_scene_ = true;
                }
            }
        }


        if (foxglove_->localizationTransformChannel().hasSinks() || foxglove_->localizationPoseChannel().hasSinks()) {
            const auto odometry = store.latest<parallax::localization::LocalizationOdometry>(parallax::core::ProductId::LocalizationOdometry);

            if (odometry && odometry->valid()) {
                const auto timestamp_ns = odometry->payload->pose.timestamp_ns;
                if (foxglove_->localizationTransformChannel().hasSinks() && timestamp_ns != last_localization_transform_timestamp_ns_) {

                    if (!publishLocalizationTransform(*odometry->payload)) return false;
                    last_localization_transform_timestamp_ns_ = timestamp_ns;
                }

                if (foxglove_->localizationPoseChannel().hasSinks() && timestamp_ns != last_localization_pose_timestamp_ns_) {

                    if (!publishLocalizationPose(*odometry->payload)) return false;
                    last_localization_pose_timestamp_ns_ = timestamp_ns;
                }
            }
        }


        if (foxglove_->localizationTrajectoryChannel().hasSinks()) {
            const auto trajectory = store.latest<parallax::localization::LocalizationTrajectory>(parallax::core::ProductId::LocalizationTrajectory);

            if (trajectory && trajectory->valid() && !trajectory->payload->poses.empty()) {

                const auto timestamp_ns = trajectory->payload->poses.back().timestamp_ns;

                if (timestamp_ns != last_localization_trajectory_timestamp_ns_) {

                    if (!publishLocalizationTrajectory(*trajectory->payload)) {
                        return false;
                    }

                    last_localization_trajectory_timestamp_ns_ = timestamp_ns;
                }
            }
        }


        if (foxglove_->localizationStateChannel().hasSinks()) {
            const auto state = store.latest<parallax::localization::LocalizationState>(parallax::core::ProductId::LocalizationState);

            if (state && state->valid()) {
                const bool changed = !has_published_localization_state_ ||
                                      state->payload->epoch != last_localization_state_epoch_ ||
                                      state->payload->consumed_frames != last_localization_state_consumed_frames_;

                if (changed) {
                    if (!publishLocalizationState(*state->payload)) return false;

                    last_localization_state_epoch_ = state->payload->epoch;
                    last_localization_state_consumed_frames_ = state->payload->consumed_frames;

                    has_published_localization_state_ = true;
                }
            }
        }

        if (foxglove_->localOccupancyChannel().hasSinks()) {
            const auto occupancy =
                store.latest<parallax::mapping::LocalOccupancyState>(
                    parallax::core::ProductId::LocalOccupancy);
            if (occupancy && occupancy->valid() && occupancy->payload &&
                occupancy->payload->gridValid()) {
                const bool changed =
                    !has_published_local_occupancy_ ||
                    occupancy->payload->localization_epoch != last_local_occupancy_epoch_ ||
                    occupancy->payload->integrated_frames != last_local_occupancy_integrated_frames_;
                if (changed) {
                    if (!publishLocalOccupancy(*occupancy)) return false;
                    last_local_occupancy_epoch_ = occupancy->payload->localization_epoch;
                    last_local_occupancy_integrated_frames_ = occupancy->payload->integrated_frames;
                    has_published_local_occupancy_ = true;
                }
            }
        }

        if (foxglove_->localizationLandmarksChannel().hasSinks()) {
            const auto landmarks = store.latest<parallax::localization::VisualLandmarkSet>(parallax::core::ProductId::LocalizationLandmarks);

            if (landmarks && landmarks->valid()) {
                if (!publishLocalizationLandmarks(*landmarks->payload)) {
                    return false;
                }
            }
        }

        return true;
    }

    bool Publisher::publishLocalOccupancy(
        const parallax::core::Product<parallax::mapping::LocalOccupancyState>& product) {

        if (!initialized_ || foxglove_ == nullptr ||
            !product.valid() || !product.payload || !product.payload->gridValid()) {
            return false;
        }

        const auto& grid = *product.payload;
        foxglove::messages::VoxelGrid message;
        message.timestamp = sourceTimestamp(product.metadata);
        message.frame_id = "localization_world";
        message.row_count = grid.row_count;
        message.column_count = grid.column_count;

        foxglove::messages::Pose pose;
        foxglove::messages::Vector3 position;
        position.x = grid.origin_m[0];
        position.y = grid.origin_m[1];
        position.z = grid.origin_m[2];
        pose.position = position;
        foxglove::messages::Quaternion orientation;
        orientation.w = 1.0;
        pose.orientation = orientation;
        message.pose = pose;

        foxglove::messages::Vector3 cell_size;
        cell_size.x = grid.voxel_size_m;
        cell_size.y = grid.voxel_size_m;
        cell_size.z = grid.voxel_size_m;
        message.cell_size = cell_size;

        constexpr std::uint32_t CellStride = 5;
        message.cell_stride = CellStride;
        message.row_stride = grid.column_count * CellStride;
        message.slice_stride = grid.row_count * message.row_stride;

        auto addField = [&message](const char* name, std::uint32_t offset) {
            foxglove::messages::PackedElementField field;
            field.name = name;
            field.offset = offset;
            field.type = foxglove::messages::PackedElementField::NumericType::UINT8;
            message.fields.push_back(std::move(field));
        };
        addField("occupancy", 0);
        addField("red", 1);
        addField("green", 2);
        addField("blue", 3);
        addField("alpha", 4);

        message.data.resize(grid.cells.size() * CellStride);
        for (std::size_t i = 0; i < grid.cells.size(); ++i) {
            const auto cell =
                static_cast<parallax::mapping::OccupancyCell>(grid.cells[i]);
            const std::size_t base = i * CellStride;
            message.data[base] = static_cast<std::byte>(grid.cells[i]);

            // Free/unknown cells remain in the product but stay invisible in
            // the default RGBA debug view. Occupied evidence is the useful
            // visual comparison against stereo scene points and LiDAR hits.
            std::uint8_t red = 0, green = 0, blue = 0, alpha = 0;
            if (cell == parallax::mapping::OccupancyCell::Occupied) {
                red = 255; green = 80; blue = 80; alpha = 220;
            }
            message.data[base + 1] = static_cast<std::byte>(red);
            message.data[base + 2] = static_cast<std::byte>(green);
            message.data[base + 3] = static_cast<std::byte>(blue);
            message.data[base + 4] = static_cast<std::byte>(alpha);
        }

        return checkFoxglove(
            foxglove_->localOccupancyChannel().log(message),
            "Failed to publish /mapping/local_occupancy");
    }

    bool Publisher::publishLeftImage(const parallax::core::Product<parallax::isp::RectifiedStereoFrame>& product,
                                    const parallax::pose::CharucoPoseResult* pose) {
        if (!initialized_ || foxglove_ == nullptr || !product.valid() || !product.payload->left.isAllocated()) return false;
        const auto& frame = *product.payload;
        if (frame.width != width_ || frame.height != height_) return false;
        const std::size_t host_pitch = static_cast<std::size_t>(width_) * 3U * sizeof(std::uint8_t);
        if (!frame.left.downloadAsync(host_rgb_, host_pitch, stream_)) return false;
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return false;

        cv::Mat image(static_cast<int>(height_), static_cast<int>(width_), CV_8UC3, host_rgb_, host_pitch);
        if (pose != nullptr && pose->pose_valid) {
            std::vector<cv::Point> polygon;
            polygon.reserve(4);
            for (const auto& point : pose->projected_plane) polygon.emplace_back(static_cast<int>(std::lround(point.x)), static_cast<int>(std::lround(point.y)));
            cv::polylines(image, polygon, true, cv::Scalar(0, 255, 0), 5, cv::LINE_AA);
            cv::circle(image, cv::Point(static_cast<int>(std::lround(pose->projected_center.x)), static_cast<int>(std::lround(pose->projected_center.y))), 8, cv::Scalar(255, 0, 0), -1);
        }

        cv::Mat bgr(static_cast<int>(height_), static_cast<int>(width_), CV_8UC3, bgr_storage_.data(), host_pitch);
        cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);
        const std::vector<int> parameters{cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
        jpeg_bytes_.clear();
        if (!cv::imencode(".jpg", bgr, jpeg_bytes_, parameters)) return false;

        foxglove::messages::CompressedImage message;
        message.timestamp = sourceTimestamp(product.metadata);
        message.frame_id = coordinate_frame_;
        message.format = "jpeg";
        message.data.resize(jpeg_bytes_.size());
        std::memcpy(message.data.data(), jpeg_bytes_.data(), jpeg_bytes_.size());
        return checkFoxglove(foxglove_->leftImageChannel().log(message), "Failed to publish /camera/left/image");
    }

    bool Publisher::publishDepth(const parallax::core::Product<parallax::isp::DepthFrame>& product,
                                 bool publish_image,
                                 bool publish_scene) {
        if (!initialized_ || foxglove_ == nullptr || !product.valid() ||
            !product.payload->depth.isAllocated() || (!publish_image && !publish_scene)) return false;

        const auto& frame = *product.payload;
        if (frame.width != width_ || frame.height != height_) {
            std::cerr << "Visualization depth dimensions changed\n";
            return false;
        }

        if (!parallax::cuda::downsampleDepthNearest(frame.depth, depth_preview_, DepthPreviewStride, stream_)) return false;

        const std::size_t host_pitch = static_cast<std::size_t>(depth_preview_width_) * sizeof(float);
        if (!depth_preview_.downloadAsync(host_depth_, host_pitch, stream_)) return false;
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return false;

        if (publish_image) {
            const std::size_t bytes = static_cast<std::size_t>(depth_preview_width_) *
                                      depth_preview_height_ * sizeof(float);
            foxglove::messages::RawImage image;
            image.timestamp = sourceTimestamp(product.metadata);
            image.frame_id = coordinate_frame_;
            image.width = depth_preview_width_;
            image.height = depth_preview_height_;
            image.encoding = "32FC1";
            image.step = depth_preview_width_ * sizeof(float);
            image.data.resize(bytes);
            std::memcpy(image.data.data(), host_depth_, bytes);
            if (!checkFoxglove(foxglove_->depthChannel().log(image), "Failed to publish /stereo/depth")) return false;
        }

        return !publish_scene || publishDepthScene(product);
    }

    bool Publisher::publishDepthScene(const parallax::core::Product<parallax::isp::DepthFrame>& product) {
        const auto points = buildDepthScenePoints(host_depth_, depth_preview_width_, depth_preview_height_,
                                                  DepthSceneSampleStride, depth_preview_projection_);

        foxglove::messages::PointCloud message;
        message.timestamp = sourceTimestamp(product.metadata);
        message.frame_id = coordinate_frame_;
        message.point_stride = 3U * sizeof(float);

        foxglove::messages::Pose pose;
        foxglove::messages::Vector3 position;
        position.x = 0.0; position.y = 0.0; position.z = 0.0;
        pose.position = position;
        foxglove::messages::Quaternion orientation;
        orientation.x = 0.0; orientation.y = 0.0; orientation.z = 0.0; orientation.w = 1.0;
        pose.orientation = orientation;
        message.pose = pose;

        using NumericType = foxglove::messages::PackedElementField::NumericType;
        foxglove::messages::PackedElementField x;
        x.name = "x"; x.offset = 0; x.type = NumericType::FLOAT32;
        foxglove::messages::PackedElementField y;
        y.name = "y"; y.offset = sizeof(float); y.type = NumericType::FLOAT32;
        foxglove::messages::PackedElementField z;
        z.name = "z"; z.offset = 2U * sizeof(float); z.type = NumericType::FLOAT32;
        message.fields = {x, y, z};

        message.data.resize(points.size() * message.point_stride);
        if (!points.empty()) std::memcpy(message.data.data(), points.data(), message.data.size());

        // Source time is deliberate: Foxglove transform history places each retained
        // cloud at the pose valid for that observation. Use ~0.75 s Decay Time in 3D.
        return checkFoxglove(foxglove_->depthSceneChannel().log(message), "Failed to publish /stereo/scene");
    }

    bool Publisher::publishRuntimeTelemetry(const std::string& json) {
        if (!initialized_ || foxglove_ == nullptr) {
            return false;
        }

        return checkFoxglove(foxglove_->runtimeTelemetryChannel().log(
                                        reinterpret_cast<const std::byte*>(json.data()),
                                        json.size()),
                                        "Failed to publish /parallax/runtime");
    }

    bool Publisher::publishDisparity(const parallax::isp::StereoMatchFrame& frame) {
        if (!initialized_ || foxglove_ == nullptr || !frame.disparity.isAllocated()) {
            return false;
        }

        const std::size_t host_pitch = static_cast<std::size_t>(frame.width) * sizeof(std::int16_t);
        if (!frame.disparity.downloadAsync(host_disparity_, host_pitch, stream_)) {
            std::cerr << "Failed to download disparity\n";
            return false;
        }

        if (cudaStreamSynchronize(stream_) != cudaSuccess) { return false; }

        const std::size_t pixels = static_cast<std::size_t>(frame.width) * frame.height;
        for (std::size_t i = 0; i < pixels; ++i) {
            disparity_float_[i] = static_cast<float>(host_disparity_[i]) / parallax::isp::StereoMatchFrame::DisparityScale;
        }

        foxglove::messages::RawImage message;

        message.frame_id = coordinate_frame_;
        message.width = frame.width;
        message.height = frame.height;
        message.encoding = "32FC1";
        message.step = frame.width * sizeof(float);

        message.data.resize(pixels * sizeof(float));

        std::memcpy( message.data.data(), disparity_float_.data(), message.data.size());

        return checkFoxglove(foxglove_->disparityChannel().log(message), "Failed to publish /stereo/disparity");
    }

    bool Publisher::publishLidarScan(const parallax::lidar::LidarScan& scan) {
        if (!initialized_ || foxglove_ == nullptr || !scan.valid() || scan.points.empty()) {
            return false;
        }

        foxglove::messages::LaserScan message;

        message.timestamp = nowTimestamp();
        message.frame_id = "lidar";

        /**
         * Parallax currently treats the RPLIDAR scan frame as the message frame
         * itself, so the scan origin is identity within that frame.
         *
         * Foxglove Pose fields are optional, but supplying an explicit identity
         * pose keeps the native LaserScan geometry unambiguous.
         */
        foxglove::messages::Pose pose;

        foxglove::messages::Vector3 position;
        position.x = 0.0;
        position.y = 0.0;
        position.z = 0.0;

        foxglove::messages::Quaternion orientation;
        orientation.x = 0.0;
        orientation.y = 0.0;
        orientation.z = 0.0;
        orientation.w = 1.0;

        pose.position = position;
        pose.orientation = orientation;
        message.pose = pose;

        /**
         * SLAMTEC and Foxglove use opposite positive-angle conventions.
         * ascendScanData() has already ordered the complete SLAMTEC scan and
         * reconstructed angular positions for no-return slots using the scan's
         * 360 / count increment.
         *
         * Keeping every slot in LidarScan therefore allows us to represent the
         * result honestly through Foxglove's equally-spaced LaserScan contract.
         */
        message.start_angle = -static_cast<double>(scan.points.back().angle_rad);
        message.end_angle = -static_cast<double>(scan.points.front().angle_rad);
        message.ranges.reserve(scan.points.size());
        message.intensities.reserve(scan.points.size());

        for (auto it = scan.points.rbegin(); it != scan.points.rend(); ++it) {
                const auto& point = *it;

                message.ranges.push_back(point.valid ? static_cast<double>(point.range_m) : 0.0);
                message.intensities.push_back(static_cast<double>(point.quality));
            }
        return checkFoxglove(foxglove_->lidarScanChannel().log(message), "Failed to publish /lidar/scan");
    }

    bool Publisher::publishDetections(const parallax::core::Product<parallax::perception::DetectionSet>& product) {
        if (!initialized_ || foxglove_ == nullptr || !product.valid() || !product.payload->valid()) {
            return false;
        }

        const auto& result = *product.payload;

        nlohmann::json message;
        message["query"] = result.query;
        message["query_revision"] = result.query_revision;
        message["detected"] = !result.empty();
        message["count"] = result.size();

        message["source"] = {{"id", sourceIdName(product.metadata.observation.source)},
                            {"sequence", product.metadata.observation.sequence}};

        message["detections"] = nlohmann::json::array();

        for (std::size_t i = 0; i < result.size(); ++i) {
            const auto& box = result.boxes[i];

            message["detections"].push_back({{"label", result.labels[i]},
                                             {"score", result.scores[i]},
                                             {"x", box.x},
                                             {"y", box.y},
                                             {"width", box.width},
                                             {"height", box.height}});
        }

        const std::string serialized = message.dump();

        return checkFoxglove(foxglove_->detectionChannel().log(
                            reinterpret_cast<const std::byte*>(serialized.data()),
                            serialized.size()), 
                            "Failed to publish /perception/detections");
    }

    bool Publisher::publishDetectionAnnotations(const parallax::core::Product<parallax::perception::DetectionSet>& product) {
        if (!initialized_ || foxglove_ == nullptr || !product.valid() || !product.payload->valid()) {
            return false;
        }

        const auto& detections = *product.payload;
        foxglove::messages::ImageAnnotations message;

        message.timestamp = sourceTimestamp(product.metadata);

        /*
        * Top-level metadata keeps the visualization's coordinate/provenance
        * contract inspectable without creating another Parallax message schema.
        */
        foxglove::messages::KeyValuePair image_space;
        image_space.key = "image_space";
        image_space.value = "rgb_left_isp";
        message.metadata.push_back(std::move(image_space));

        foxglove::messages::KeyValuePair source_sequence;
        source_sequence.key = "source_sequence";
        source_sequence.value = std::to_string(product.metadata.observation.sequence);
        message.metadata.push_back(std::move(source_sequence));

        foxglove::messages::KeyValuePair query_revision;
        query_revision.key = "query_revision";
        query_revision.value = std::to_string(detections.query_revision);
        message.metadata.push_back(std::move(query_revision));

        message.points.reserve(detections.size());
        message.texts.reserve(detections.size());

        for (std::size_t i = 0; i < detections.size(); ++i) {
            const auto& box = detections.boxes[i];

            if (!std::isfinite(box.x) || !std::isfinite(box.y)
                || !std::isfinite(box.width) || !std::isfinite(box.height)
                || box.width <= 0.0F || box.height <= 0.0F) {

                continue;
            }

            const double x0 = static_cast<double>(box.x);
            const double y0 = static_cast<double>(box.y);
            const double x1 = static_cast<double>(box.x + box.width);
            const double y1 = static_cast<double>(box.y + box.height);

            foxglove::messages::PointsAnnotation rectangle;

            rectangle.type = foxglove::messages::PointsAnnotation::PointsAnnotationType::LINE_LOOP;
            rectangle.thickness = 3.0;
            rectangle.points.reserve(4);

            foxglove::messages::Point2 top_left;
            top_left.x = x0;
            top_left.y = y0;

            foxglove::messages::Point2 top_right;
            top_right.x = x1;
            top_right.y = y0;

            foxglove::messages::Point2 bottom_right;
            bottom_right.x = x1;
            bottom_right.y = y1;

            foxglove::messages::Point2 bottom_left;
            bottom_left.x = x0;
            bottom_left.y = y1;

            rectangle.points.push_back(top_left);
            rectangle.points.push_back(top_right);
            rectangle.points.push_back(bottom_right);
            rectangle.points.push_back(bottom_left);

            foxglove::messages::Color outline;
            outline.r = 0.0;
            outline.g = 1.0;
            outline.b = 0.0;
            outline.a = 1.0;

            rectangle.outline_color = outline;

            message.points.push_back(std::move(rectangle));

            foxglove::messages::TextAnnotation label;

            /*
            * Foxglove defines TextAnnotation::position as the bottom-left text
            * origin in image coordinates.
            *
            * Put the label immediately above the detector box where possible.
            */
            foxglove::messages::Point2 text_position;
            text_position.x = x0;
            text_position.y = std::max(18.0, y0 - 4.0);

            label.position = text_position;
            label.font_size = 18.0;

            std::ostringstream text;
            text << detections.query << ' '
                 << std::fixed << std::setprecision(2)
                 << detections.scores[i];

            label.text = text.str();

            foxglove::messages::Color text_color;
            text_color.r = 1.0;
            text_color.g = 1.0;
            text_color.b = 1.0;
            text_color.a = 1.0;

            label.text_color = text_color;

            foxglove::messages::Color background;
            background.r = 0.0;
            background.g = 0.0;
            background.b = 0.0;
            background.a = 0.65;

            label.background_color = background;
            message.texts.push_back(std::move(label));
        }
        return checkFoxglove(foxglove_->detectionAnnotationsChannel().log(message), "Failed to publish /perception/annotations");
    }

    bool Publisher::publishSegmentationMask(const parallax::core::Product<parallax::perception::SegmentationMask>& product) {
        if (!initialized_ || foxglove_ == nullptr || !product.valid() || !product.payload || !product.payload->valid()) {
            return false;
        }

        const auto& mask = *product.payload;
        if (mask.representation != parallax::perception::MaskRepresentation::CudaDevice ||
            mask.layout != parallax::perception::MaskLayout::RowMajor) {

            return false;
        }

        if (mask.width != width_ || mask.height != height_) {
            std::cerr << "Visualization segmentation dimensions changed\n";
            return false;
        }

        const std::size_t host_pitch = static_cast<std::size_t>(mask.width);
        if (cudaMemcpy2DAsync(host_segmentation_mask_,
                              host_pitch,
                              mask.storage.get(),
                              mask.pitch_bytes,
                              host_pitch,
                              mask.height,
                              cudaMemcpyDeviceToHost,
                              stream_) != cudaSuccess) {

            std::cerr << "Failed to download segmentation mask\n";

            return false;
        }

        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            std::cerr << "Failed to synchronize segmentation mask download\n";
            return false;
        }

        foxglove::messages::RawImage message;

        message.timestamp = sourceTimestamp(product.metadata);
        message.frame_id = coordinate_frame_;
        message.width = mask.width;
        message.height = mask.height;
        message.encoding = "mono8";
        message.step = mask.width;

        const std::size_t bytes = static_cast<std::size_t>(mask.width) * mask.height;
        message.data.resize(bytes);

        std::memcpy(message.data.data(), host_segmentation_mask_, bytes);
        return checkFoxglove(foxglove_->segmentationMaskChannel().log(message), "Failed to publish /perception/segmentation");
    }
    
    bool Publisher::publishTrackAnnotations(const parallax::core::Product<parallax::tracking::Track2D>& product) {
        if (!initialized_ ||
            foxglove_ == nullptr ||
            !product.valid() ||
            !product.payload ||
            !product.payload->valid()) {

            return false;
        }

        const auto& track = *product.payload;
        const auto& box = track.box;

        if (!std::isfinite(box.x) ||
            !std::isfinite(box.y) ||
            !std::isfinite(box.width) ||
            !std::isfinite(box.height) ||
            box.width <= 0.0F ||
            box.height <= 0.0F) {

            return false;
        }

        foxglove::messages::ImageAnnotations message;
        message.timestamp = sourceTimestamp(product.metadata);

        // Track boxes stay in RgbLeft pixels.
        foxglove::messages::KeyValuePair image_space;
        image_space.key = "image_space";
        image_space.value = "rgb_left_isp";
        message.metadata.push_back(std::move(image_space));

        foxglove::messages::KeyValuePair source_sequence;
        source_sequence.key = "source_sequence";
        source_sequence.value = std::to_string(product.metadata.observation.sequence);

        message.metadata.push_back(std::move(source_sequence));

        const double x0 = box.x;
        const double y0 = box.y;
        const double x1 = box.x + box.width;
        const double y1 = box.y + box.height;

        foxglove::messages::PointsAnnotation rectangle;
        rectangle.type = foxglove::messages::PointsAnnotation::PointsAnnotationType::LINE_LOOP;

        rectangle.thickness = 3.0;

        rectangle.points = {{x0, y0},
                            {x1, y0},
                            {x1, y1},
                            {x0, y1}};

        foxglove::messages::Color outline;
        outline.r = 1.0;
        outline.g = 0.7;
        outline.b = 0.0;
        outline.a = 1.0;
        rectangle.outline_color = outline;

        message.points.push_back(std::move(rectangle));

        foxglove::messages::TextAnnotation label;

        foxglove::messages::Point2 position;
        position.x = x0;
        position.y = std::max(18.0, y0 - 4.0);

        label.position = position;
        label.font_size = 18.0;

        std::ostringstream text;
        text << "track " << track.track_id
            << ' ' << track.target_query
            << ' ' << std::fixed
            << std::setprecision(2)
            << track.quality;

        label.text = text.str();

        foxglove::messages::Color text_color;
        text_color.r = 1.0;
        text_color.g = 1.0;
        text_color.b = 1.0;
        text_color.a = 1.0;
        label.text_color = text_color;

        foxglove::messages::Color background;
        background.r = 0.0;
        background.g = 0.0;
        background.b = 0.0;
        background.a = 0.65;
        label.background_color = background;

        message.texts.push_back(std::move(label));

        return checkFoxglove(foxglove_->trackAnnotationsChannel().log(message), "Failed to publish /tracking/annotations");
    }

    bool Publisher::publishObjectDepthAnnotations(const parallax::core::Product<parallax::perception::Object3DSet>& product) {
        if (!initialized_ || foxglove_ == nullptr || !product.valid() || !product.payload || !product.payload->valid()) {
            return false;
        }

        foxglove::messages::ImageAnnotations message;
        message.timestamp = sourceTimestamp(product.metadata);

        foxglove::messages::KeyValuePair image_space;
        image_space.key = "image_space";
        image_space.value = "rectified_left";
        message.metadata.push_back(std::move(image_space));

        for (const auto& object : product.payload->objects) {
            if (!object.valid() ||
                object.depth_image_space != parallax::perception::ImageSpace::RectifiedLeft ||
                object.depth_roi.width <= 0.0F || object.depth_roi.height <= 0.0F) {
                continue;
            }

            const auto& roi = object.depth_roi;

            foxglove::messages::PointsAnnotation rectangle;
            rectangle.type = foxglove::messages::PointsAnnotation::PointsAnnotationType::LINE_LOOP;
            rectangle.thickness = 3.0;

            rectangle.points = {{roi.x, roi.y},
                                {roi.x + roi.width, roi.y},
                                {roi.x + roi.width, roi.y + roi.height},
                                {roi.x, roi.y + roi.height}};

            foxglove::messages::Color outline;
            outline.r = 1.0;
            outline.g = 1.0;
            outline.b = 0.0;
            outline.a = 1.0;

            rectangle.outline_color = outline;
            message.points.push_back(std::move(rectangle));

            const std::string distance = formatDepth(object.depth_m);

            if (distance.empty()) continue;

            foxglove::messages::TextAnnotation label;
            foxglove::messages::Point2 position;
            position.x = roi.x;
            position.y = std::max(18.0, static_cast<double>(roi.y - 4.0F));

            label.position = position;
            label.font_size = 18.0;
            label.text = object.label + " · " + distance;

            foxglove::messages::Color text;
            text.r = 1.0;
            text.g = 1.0;
            text.b = 1.0;
            text.a = 1.0;

            label.text_color = text;

            foxglove::messages::Color background;
            background.r = 0.0;
            background.g = 0.0;
            background.b = 0.0;
            background.a = 0.65;

            label.background_color = background;

            message.texts.push_back(std::move(label));
        }

        return checkFoxglove(foxglove_->objectDepthAnnotationsChannel().log(message), "Failed to publish /perception/depth/annotations");
    }

    bool Publisher::publishObject3DScene(const parallax::core::Product<parallax::perception::Object3DSet>& product) {
        if (!initialized_ || foxglove_ == nullptr || !product.valid() || !product.payload || !product.payload->valid()) {
            return false;
        }

        return publishObject3DScene(*product.payload,
                                    sourceTimestamp(product.metadata),
                                    foxglove_->object3DSceneChannel(),
                                    "Failed to publish /perception/objects3d");
    }

    bool Publisher::publishTrackedObject3DScene(
        const parallax::core::Product<parallax::perception::Object3DSet>& product) {

        if (!initialized_ || foxglove_ == nullptr ||
            !product.valid() || !product.payload || !product.payload->valid()) {
            return false;
        }

        return publishObject3DScene(*product.payload,
                                    sourceTimestamp(product.metadata),
                                    foxglove_->trackedObject3DSceneChannel(),
                                    "Failed to publish /tracking/objects3d");
    }

    bool Publisher::publishLocalizedObject3DScene(const parallax::core::Product<parallax::perception::LocalizedSpatialObservation>& product) {
        if (!initialized_ || foxglove_ == nullptr || !product.valid() || !product.payload || !product.payload->valid()) {
            return false;
        }

        return publishObject3DScene(product.payload->objects,
                                    sourceTimestamp(product.metadata),
                                    foxglove_->localizedObject3DSceneChannel(),
                                    "Failed to publish /localization/objects3d");
    }

    bool Publisher::publishObject3DScene(const parallax::perception::Object3DSet& objects,
                                         const foxglove::messages::Timestamp& timestamp,
                                         foxglove::messages::SceneUpdateChannel& channel,
                                         const char* error_message) {
        if (!initialized_ || foxglove_ == nullptr || !objects.valid()) return false;

        foxglove::messages::SceneUpdate update;
        update.entities.reserve(objects.objects.size());

        for (std::size_t i = 0; i < objects.objects.size(); ++i) {
            const auto& object = objects.objects[i];
            if (!object.valid()) continue;

            foxglove::messages::SceneEntity entity;
            entity.timestamp = timestamp;
            entity.frame_id = object.coordinate_frame;

            /*
            * Persistent tracks use stable track identity.
            * Frame-local detections use stable slot identity so each SceneUpdate
            * replaces the previous visualization instead of accumulating history.
            */
            if (object.persistent()) {
                entity.id = "track_" + std::to_string(object.track_id);
            } else {
                entity.id = "object_" + std::to_string(i);
            }

            entity.frame_locked = true;

            /*
            * The sphere marks the representative metric position produced by
            * stereo ROI association. It does not imply object volume.
            */
            foxglove::messages::SpherePrimitive marker;
            foxglove::messages::Pose marker_pose;
            foxglove::messages::Vector3 position;

            position.x = object.position_m[0];
            position.y = object.position_m[1];
            position.z = object.position_m[2];

            marker_pose.position = position;

            foxglove::messages::Quaternion orientation;
            orientation.w = 1.0;

            marker_pose.orientation = orientation;
            marker.pose = marker_pose;

            foxglove::messages::Vector3 size;
            size.x = 0.04;
            size.y = 0.04;
            size.z = 0.04;

            marker.size = size;

            foxglove::messages::Color marker_color;
            marker_color.r = 1.0;
            marker_color.g = 1.0;
            marker_color.b = 0.0;
            marker_color.a = 1.0;

            marker.color = marker_color;
            entity.spheres.push_back(std::move(marker));

            /*
            * Draw only the planar image-supported rectangle. This does not
            * represent measured object width, height, or physical volume.
            */
            if (object.geometry == parallax::perception::Object3DGeometry::ImageSupportedGeometry) {
                foxglove::messages::LinePrimitive rectangle;
                rectangle.type = foxglove::messages::LinePrimitive::LineType::LINE_LOOP;

                rectangle.thickness = 2.0;
                rectangle.scale_invariant = true;
                rectangle.points.reserve(4);

                for (const auto& corner : object.image_supported_corners_m) {
                    foxglove::messages::Point3 point;

                    point.x = corner[0];
                    point.y = corner[1];
                    point.z = corner[2];

                    rectangle.points.push_back(point);
                }

                foxglove::messages::Color line_color;
                line_color.r = 0.0;
                line_color.g = 1.0;
                line_color.b = 0.0;
                line_color.a = 1.0;

                rectangle.color = line_color;
                entity.lines.push_back(std::move(rectangle));
            }

            if ((object.geometry == parallax::perception::Object3DGeometry::Surface ||
                 object.geometry == parallax::perception::Object3DGeometry::ObservedExtent) &&
                !object.surface_points_m.empty()) {
                constexpr std::size_t MaxVisibleSurfacePoints = 64;
                const std::size_t visible = std::min(object.surface_points_m.size(), MaxVisibleSurfacePoints);
                
                entity.spheres.reserve(entity.spheres.size() + visible);

                for (std::size_t point_index = 0; point_index < visible; ++point_index) {
                    const auto& sample = object.surface_points_m[point_index];
                    
                    foxglove::messages::SpherePrimitive point;
                    foxglove::messages::Pose pose;
                    
                    foxglove::messages::Vector3 position;
                    position.x = sample[0];
                    position.y = sample[1];
                    position.z = sample[2];
                    pose.position = position;
                    
                    foxglove::messages::Quaternion orientation;
                    orientation.w = 1.0;
                    pose.orientation = orientation;
                    point.pose = pose;
                    
                    foxglove::messages::Vector3 size;
                    size.x = 0.012;
                    size.y = 0.012;
                    size.z = 0.012;
                    point.size = size;
                    
                    foxglove::messages::Color color;
                    color.r = 0.0;
                    color.g = 0.75;
                    color.b = 1.0;
                    color.a = 0.8;

                    point.color = color;
                    entity.spheres.push_back(std::move(point));
                }
            }

            if (object.geometry == parallax::perception::Object3DGeometry::ObservedExtent) {
                /*
                 * ObservedExtent bounds only stereo-supported visible samples.
                 * This translucent box is a measurement envelope, not a claim
                 * about hidden/back-side physical dimensions.
                 *
                 * The bounds are axis-aligned in object.coordinate_frame, so
                 * identity orientation is intentional. A future oriented extent
                 * would require a different geometry contract.
                 */
                foxglove::messages::CubePrimitive extent;
                foxglove::messages::Pose extent_pose;

                foxglove::messages::Vector3 extent_position;
                extent_position.x = object.observed_extent_center_m[0];
                extent_position.y = object.observed_extent_center_m[1];
                extent_position.z = object.observed_extent_center_m[2];
                extent_pose.position = extent_position;

                foxglove::messages::Quaternion extent_orientation;
                extent_orientation.w = 1.0;
                extent_pose.orientation = extent_orientation;
                extent.pose = extent_pose;

                foxglove::messages::Vector3 extent_size;
                extent_size.x = object.observed_extent_size_m[0];
                extent_size.y = object.observed_extent_size_m[1];
                extent_size.z = object.observed_extent_size_m[2];
                extent.size = extent_size;

                foxglove::messages::Color extent_color;
                extent_color.r = 0.0;
                extent_color.g = 0.75;
                extent_color.b = 1.0;
                extent_color.a = 0.18;
                extent.color = extent_color;

                entity.cubes.push_back(std::move(extent));

                foxglove::messages::KeyValuePair support;
                support.key = "observed_extent_support";
                support.value = std::to_string(object.observed_extent_support);
                entity.metadata.push_back(std::move(support));
            }

            const std::string distance = formatDepthForDisplay(objectDisplayDistance(object));

            // Billboard text is presentation-only. It follows the measured point
            // and remains readable while the Foxglove 3D camera is moved.
            foxglove::messages::TextPrimitive text;
            foxglove::messages::Pose text_pose;
            foxglove::messages::Vector3 text_position;

            text_position.x = object.position_m[0];
            text_position.y = object.position_m[1];
            text_position.z = object.position_m[2] + 0.05;

            text_pose.position = text_position;

            foxglove::messages::Quaternion text_orientation;
            text_orientation.w = 1.0;

            text_pose.orientation = text_orientation;
            text.pose = text_pose;
            text.billboard = true;
            text.scale_invariant = true;
            text.font_size = 14.0;

            foxglove::messages::Color text_color;
            text_color.r = 1.0;
            text_color.g = 1.0;
            text_color.b = 1.0;
            text_color.a = 1.0;
            text.color = text_color;

            const std::string metric_source = objectMetricSourceName(object.method);
            text.text = distance.empty()
                            ? object.label
                            : object.label + " · " + distance + " · " + metric_source;

            entity.texts.push_back(std::move(text));

            // Keep semantic and metric values available as inspectable entity
            // metadata in addition to their rendered representation.
            foxglove::messages::KeyValuePair label;
            label.key = "label";
            label.value = object.label;

            entity.metadata.push_back(std::move(label));
            if (!distance.empty()) {
                foxglove::messages::KeyValuePair depth;
                depth.key = "distance";
                depth.value = distance;
                entity.metadata.push_back(std::move(depth));
            }

            foxglove::messages::KeyValuePair metric_source_metadata;
            metric_source_metadata.key = "metric_source";
            metric_source_metadata.value = metric_source;
            entity.metadata.push_back(std::move(metric_source_metadata));

            update.entities.push_back(std::move(entity));
        }

        return checkFoxglove(channel.log(update), error_message);
    }

    std::string formatDepthForDisplay(float depth_m) {
        if (!std::isfinite(depth_m) || depth_m <= 0.0F) return {};

        constexpr float MetersToFeet = 3.280839895F;
        constexpr float FeetToInches = 12.0F;
        const float feet = depth_m * MetersToFeet;

        std::ostringstream stream;
        stream << std::fixed;

        if (feet < 1.0F) {
            stream << std::setprecision(1) << feet * FeetToInches << " in";
        } else if (feet < 3.0F) {
            stream << std::setprecision(1) << feet << " ft";
        } else {
            stream << std::setprecision(2) << depth_m << " m";
        }

        return stream.str();
    }

    const char* localizationStateName(parallax::localization::LocalizationTrackingState state) noexcept {
        using State = parallax::localization::LocalizationTrackingState;

        switch (state) {
            case State::Tracking:
                return "tracking";

            case State::Lost:
                return "lost";

            case State::Uninitialized:
            default:
                return "uninitialized";
        }
    }

    bool Publisher::publishLocalizationTransform(const parallax::localization::LocalizationOdometry& odometry) {
        if (!initialized_ || foxglove_ == nullptr) return false;

        const auto& pose = odometry.pose;

        foxglove::messages::FrameTransform message;
        message.timestamp = nowTimestamp();
        message.parent_frame_id = "localization_world";
        message.child_frame_id = "stereo_body";

        foxglove::messages::Vector3 translation;
        translation.x = pose.translation_m[0];
        translation.y = pose.translation_m[1];
        translation.z = pose.translation_m[2];
        message.translation = translation;

        foxglove::messages::Quaternion rotation;
        rotation.x = pose.rotation_xyzw[0];
        rotation.y = pose.rotation_xyzw[1];
        rotation.z = pose.rotation_xyzw[2];
        rotation.w = pose.rotation_xyzw[3];
        message.rotation = rotation;

        return checkFoxglove(foxglove_->localizationTransformChannel().log(message), "Failed to publish /localization/transform");
    }

    bool Publisher::publishLocalizationPose(const parallax::localization::LocalizationOdometry& odometry) {
        if (!initialized_ || foxglove_ == nullptr) return false;

        const auto& localization = odometry.pose;

        foxglove::messages::PoseInFrame message;
        message.timestamp = nowTimestamp();
        message.frame_id = "localization_world";

        foxglove::messages::Pose pose;

        foxglove::messages::Vector3 position;
        position.x = localization.translation_m[0];
        position.y = localization.translation_m[1];
        position.z = localization.translation_m[2];
        pose.position = position;

        foxglove::messages::Quaternion orientation;
        orientation.x = localization.rotation_xyzw[0];
        orientation.y = localization.rotation_xyzw[1];
        orientation.z = localization.rotation_xyzw[2];
        orientation.w = localization.rotation_xyzw[3];
        pose.orientation = orientation;

        message.pose = pose;

        return checkFoxglove(foxglove_->localizationPoseChannel().log(message), "Failed to publish /localization/pose");
    }

    bool Publisher::publishLocalizationTrajectory(const parallax::localization::LocalizationTrajectory& trajectory) {
        if (!initialized_ || foxglove_ == nullptr) return false;

        foxglove::messages::SceneUpdate update;
        foxglove::messages::SceneEntity entity;

        entity.timestamp = nowTimestamp();
        entity.frame_id = "localization_world";
        entity.id = "localization_trajectory_" + std::to_string(trajectory.epoch);

        entity.frame_locked = true;

        if (trajectory.poses.size() >= 2) {
            foxglove::messages::LinePrimitive path;
            path.type = foxglove::messages::LinePrimitive::LineType::LINE_STRIP;

            path.thickness = 3.0;
            path.scale_invariant = true;
            path.points.reserve(trajectory.poses.size());

            for (const auto& pose : trajectory.poses) {
                foxglove::messages::Point3 point;
                point.x = pose.translation_m[0];
                point.y = pose.translation_m[1];
                point.z = pose.translation_m[2];

                path.points.push_back(point);
            }

            foxglove::messages::Color color;
            color.r = 0.0;
            color.g = 0.8;
            color.b = 1.0;
            color.a = 1.0;

            path.color = color;
            entity.lines.push_back(std::move(path));
        }

        foxglove::messages::KeyValuePair epoch;
        epoch.key = "localization_epoch";
        epoch.value = std::to_string(trajectory.epoch);
        entity.metadata.push_back(std::move(epoch));

        update.entities.push_back(std::move(entity));

        return checkFoxglove(foxglove_->localizationTrajectoryChannel().log(update), "Failed to publish /localization/trajectory");
    }

    bool Publisher::publishLocalizationState(const parallax::localization::LocalizationState& state) {
        if (!initialized_ || foxglove_ == nullptr) return false;

        nlohmann::json json{{"tracking", localizationStateName(state.tracking)},
                            {"epoch", state.epoch},
                            {"consumed_frames", state.consumed_frames},
                            {"input_gaps", state.input_gaps},
                            {"session_resets", state.session_resets}};

        const std::string serialized = json.dump();
        return checkFoxglove(foxglove_->localizationStateChannel().log(reinterpret_cast<const std::byte*>(serialized.data()),
                                                                                                            serialized.size()),
                                                                                                            "Failed to publish /localization/state");
    }

    bool Publisher::publishLocalizationLandmarks(const parallax::localization::VisualLandmarkSet& landmarks) {
        if (!initialized_ || foxglove_ == nullptr) return false;

        foxglove::messages::SceneUpdate update;

        foxglove::messages::SceneEntity entity;
        entity.timestamp = nowTimestamp();
        entity.frame_id = "localization_world";
        entity.id = "cuvslam_landmarks";
        entity.frame_locked = true;

        entity.spheres.reserve(landmarks.landmarks.size());

        for (const auto& landmark : landmarks.landmarks) {
            foxglove::messages::SpherePrimitive point;
            foxglove::messages::Pose pose;

            foxglove::messages::Vector3 position;
            position.x = landmark.position_m[0];
            position.y = landmark.position_m[1];
            position.z = landmark.position_m[2];

            pose.position = position;

            foxglove::messages::Quaternion orientation;
            orientation.w = 1.0;
            pose.orientation = orientation;

            point.pose = pose;

            foxglove::messages::Vector3 size;
            size.x = 0.01;
            size.y = 0.01;
            size.z = 0.01;
            point.size = size;

            foxglove::messages::Color color;
            color.r = 0.8;
            color.g = 0.2;
            color.b = 0.8;
            color.a = 1.0;
            point.color = color;

            entity.spheres.push_back(std::move(point));
        }

        update.entities.push_back(std::move(entity));

        return checkFoxglove(foxglove_->localizationLandmarksChannel().log(update), "Failed to publish /localization/landmarks");
    }

    void Publisher::shutdown() {

        if (host_rgb_ != nullptr) {
            cudaFreeHost(host_rgb_);
            host_rgb_ = nullptr;
        }

        if (host_depth_ != nullptr) {
            cudaFreeHost(host_depth_);
            host_depth_ = nullptr;
        }

        depth_preview_.release();
        depth_preview_width_ = 0;
        depth_preview_height_ = 0;
        last_depth_publish_ = {};

        if (host_disparity_ != nullptr) {
            cudaFreeHost(host_disparity_);
            host_disparity_ = nullptr;
        }

        if (host_segmentation_mask_ != nullptr) {
            cudaFreeHost(host_segmentation_mask_);
            host_segmentation_mask_ = nullptr;
        }
        
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        
        disparity_float_.clear();

        last_segmentation_observation_ = {};
        last_segmentation_query_revision_ = 0;
        has_published_segmentation_ = false;

        last_segmentation_scene_observation_ = {};
        last_segmentation_scene_revision_ = 0;
        has_published_segmentation_scene_ = false;
        last_tracked_object_scene_method_ = parallax::perception::Object3DMethod::Unknown;

        width_ = 0;
        height_ = 0;
        fps_ = 0;

        initialized_ = false;
        foxglove_ = nullptr;
    }
}
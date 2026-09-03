#include <parallax/visualization/publisher.hpp>
#include <parallax/pose/charuco_pose.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <parallax/perception/detection.hpp>

#include <algorithm>
#include <cstring>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
#include <chrono>
#include <array>
#include <iomanip>
#include <cmath>

namespace parallax::visualization {
    namespace {
        constexpr const char* kFrameId = "camera_left_optical";

        bool checkFoxglove(const foxglove::FoxgloveError& error, const char* message) {
            if (error != foxglove::FoxgloveError::Ok) {
                std::cerr << message << ": "
                          << foxglove::strerror(error) << '\n';

                return false;
            }
            return true;
        }

        foxglove::messages::Timestamp nowTimestamp() {
            const auto now = std::chrono::system_clock::now();
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                now.time_since_epoch()).count();

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
    }

    Publisher::~Publisher() { shutdown(); }

    bool Publisher::initialize(FoxgloveServer& foxglove, std::uint32_t width, std::uint32_t height, std::uint32_t fps) {
        if (initialized_) return true;

        if (width == 0 || height == 0 || fps == 0) {
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

        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            std::cerr << "Failed to create visualization CUDA stream\n";
            shutdown();
            return false;
        }

        const std::size_t pixels = static_cast<std::size_t>(width_) * height_;
        const std::size_t rgb_bytes = pixels * 3 * sizeof(std::uint8_t);

        const std::size_t disparity_bytes = pixels * sizeof(std::int16_t);
        const std::size_t depth_bytes = pixels * sizeof(float);
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

        if (cudaMallocHost(reinterpret_cast<void**>(&host_depth_), depth_bytes) != cudaSuccess) {
            std::cerr << "Failed to allocate depth staging buffer\n";
            shutdown();
            return false;
        }

        disparity_float_.resize(pixels);

        if (!video_encoder_.initialize(width_, height_, fps_)) {
            std::cerr << "Failed to initialize video encoder\n";
            shutdown();
            return false;
        }

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

        message.frame_id = kFrameId;

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
                const parallax::pose::CharucoPoseResult* overlay = nullptr;
                const auto marker = store.latest<parallax::pose::CharucoPoseResult>(
                                                parallax::core::ProductId::MarkerDepth);

                if (marker && marker->valid() && parallax::core::same_source_observation(
                                                                *marker,
                                                                rgb->metadata.observation)) {

                    overlay = marker->payload.get();
                }

                if (!rgb->completion.valid()) return false;

                if (rgb->completion.requires_wait() && !wait_for_host(rgb->completion)) {
                    return false;
                }

                if (!publishLeftImage(*rgb->payload, overlay)) return false;
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

            if (stereo && stereo->valid()) {
                if (!stereo->completion.valid()) return false;
                if (stereo->completion.requires_wait() && !wait_for_host(stereo->completion)) {
                    return false;
                }

                if (disparity_requested && !publishDisparity(*stereo->payload)) {
                    return false;
                }
            }
        }

        if (foxglove_->depthChannel().hasSinks()) {
            const auto depth = store.latest<parallax::isp::DepthFrame>(parallax::core::ProductId::Depth);

            if (depth && depth->valid()) {
                if (!depth->completion.valid()) return false;

                if (depth->completion.requires_wait() && !wait_for_host(depth->completion)) {
                    return false;
                }

                if (!publishDepth(*depth->payload)) return false;
            }
        }

        if (foxglove_->lidarScanChannel().hasSinks()) {
            const auto lidar = store.latest<parallax::lidar::LidarScan>(parallax::core::ProductId::LidarScan);

            if (lidar && lidar->valid() && !publishLidarScan(*lidar->payload)) {
                return false;
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

        return true;
    }


    bool Publisher::publishLeftImage(const parallax::isp::RectifiedStereoFrame& frame,
                                     const parallax::pose::CharucoPoseResult* pose) {

        if (!initialized_ || foxglove_ == nullptr || !frame.left.isAllocated()) {
            return false;
        }

        if (frame.width != width_ || frame.height != height_) {
            std::cerr << "Visualization RGB dimensions changed\n";
            return false;
        }

        const std::size_t host_pitch = static_cast<std::size_t>(width_) *
                                       parallax::isp::RectifiedStereoFrame::Channels *
                                       sizeof(std::uint8_t);

        if (!frame.left.downloadAsync(host_rgb_, host_pitch, stream_)) {
            std::cerr << "Failed to download left RGB frame\n";
            return false;
        }

        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            std::cerr << "Failed to synchronize RGB download\n";
            return false;
        }

        cv::Mat image(static_cast<int>(height_), static_cast<int>(width_), CV_8UC3, host_rgb_, host_pitch);

        if (pose != nullptr && pose->pose_valid) {
            std::vector<cv::Point> polygon;
            polygon.reserve(4);

            for (const auto& p : pose->projected_plane) {
                polygon.emplace_back(static_cast<int>(std::lround(p.x)),
                                     static_cast<int>(std::lround(p.y)));
            }
            cv::polylines(image, polygon, true, cv::Scalar(0, 255, 0), 5, cv::LINE_AA);

            cv::circle(image, 
                      cv::Point(static_cast<int>(std::lround(pose->projected_center.x)),
                                static_cast<int>(std::lround(pose->projected_center.y))),
                      8,
                      cv::Scalar(255, 0, 0),
                      -1);
        }
    

        const std::size_t rgb_bytes = host_pitch * height_;

        if (!video_encoder_.encode(host_rgb_, rgb_bytes, encoded_video_)) {
            return false;
        }

        foxglove::messages::CompressedVideo message;

        message.timestamp = nowTimestamp();
        message.frame_id = kFrameId;
        message.format = "h264";
        message.data = encoded_video_;

        return checkFoxglove(foxglove_->leftImageChannel().log(message), "Failed to publish /camera/left/image");
    }

    bool Publisher::publishDepth(const parallax::isp::DepthFrame& frame) {
        if (!initialized_ || foxglove_ == nullptr || !frame.depth.isAllocated()) {
            return false;
        }

        if (frame.width != width_ || frame.height != height_) {
            std::cerr << "Visualization depth dimensions changed\n";
            return false;
        }

        const std::size_t host_pitch = static_cast<std::size_t>(width_) * sizeof(float);

        if (!frame.depth.downloadAsync(host_depth_, host_pitch, stream_)) {
            std::cerr << "Failed to download left RGB frame\n";
            return false;
        }

        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            std::cerr << "Failed to synchronize RGB download\n";
            return false;
        }

        const std::size_t bytes = static_cast<std::size_t>(frame.width) * frame.height * sizeof(float);

        foxglove::messages::RawImage message;

        message.timestamp = nowTimestamp();
        message.frame_id = kFrameId;
        message.width = frame.width;
        message.height = frame.height;
        message.encoding = "32FC1";
        message.step = frame.width * sizeof(float);

        message.data.resize(bytes);

        std::memcpy(message.data.data(), host_depth_, bytes);

        return checkFoxglove(foxglove_->depthChannel().log(message), "Failed to publish /stereo/depth");
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
            disparity_float_[i] = static_cast<float>(host_disparity_[i]) /
                                    parallax::isp::StereoMatchFrame::DisparityScale;
        }

        foxglove::messages::RawImage message;

        message.frame_id = kFrameId;
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

        /*
        * ProductMetadata uses the application's steady-clock observation domain,
        * not Unix epoch time, so it cannot be copied into Foxglove Timestamp.
        *
        * Use publication time here rather than manufacturing a false conversion.
        * SourceObservation remains available in the machine-readable DetectionSet
        * channel for exact graph provenance.
        */
        message.timestamp = nowTimestamp();

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

        message.timestamp = nowTimestamp();
        message.frame_id = kFrameId;
        message.width = mask.width;
        message.height = mask.height;
        message.encoding = "mono8";
        message.step = mask.width;

        const std::size_t bytes = static_cast<std::size_t>(mask.width) * mask.height;
        message.data.resize(bytes);

        std::memcpy(message.data.data(), host_segmentation_mask_, bytes);
        return checkFoxglove(foxglove_->segmentationMaskChannel().log(message), "Failed to publish /perception/segmentation");
    }
    

    void Publisher::shutdown() {
        video_encoder_.shutdown();

        if (host_rgb_ != nullptr) {
            cudaFreeHost(host_rgb_);
            host_rgb_ = nullptr;
        }

        if (host_depth_ != nullptr) {
            cudaFreeHost(host_depth_);
            host_depth_ = nullptr;
        }

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
        encoded_video_.clear();

        last_segmentation_observation_ = {};
        last_segmentation_query_revision_ = 0;
        has_published_segmentation_ = false;

        width_ = 0;
        height_ = 0;
        fps_ = 0;

        initialized_ = false;
        foxglove_ = nullptr;
    }
}
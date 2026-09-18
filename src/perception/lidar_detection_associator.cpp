#include <parallax/perception/lidar_detection_associator.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace parallax::perception {
    namespace {
        [[nodiscard]] bool finiteArray(const std::array<double, 9>& values) noexcept {
            return std::all_of(values.begin(), values.end(), [](double value) {
                return std::isfinite(value);
            });
        }

        [[nodiscard]] bool finiteArray(const std::array<double, 3>& values) noexcept {
            return std::all_of(values.begin(), values.end(), [](double value) {
                return std::isfinite(value);
            });
        }

        [[nodiscard]] bool quaternionToMatrix(const std::array<double, 4>& q,
                                              std::array<double, 9>& rotation) noexcept {
            double x = q[0];
            double y = q[1];
            double z = q[2];
            double w = q[3];

            const double norm = std::sqrt(x * x + y * y + z * z + w * w);
            if (!std::isfinite(norm) || norm <= 1.0e-9) return false;

            x /= norm;
            y /= norm;
            z /= norm;
            w /= norm;

            rotation = {
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),

                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),

                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y)
            };

            return true;
        }

        [[nodiscard]] std::array<double, 9> transpose(const std::array<double, 9>& m) noexcept {
            return {m[0], m[3], m[6],
                    m[1], m[4], m[7],
                    m[2], m[5], m[8]};
        }

        [[nodiscard]] std::array<double, 9> multiply(const std::array<double, 9>& a,
                                                     const std::array<double, 9>& b) noexcept {
            std::array<double, 9> out{};

            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t column = 0; column < 3; ++column) {
                    out[row * 3 + column] =
                        a[row * 3 + 0] * b[0 * 3 + column] +
                        a[row * 3 + 1] * b[1 * 3 + column] +
                        a[row * 3 + 2] * b[2 * 3 + column];
                }
            }

            return out;
        }

        [[nodiscard]] std::array<double, 3> transformVector(const std::array<double, 9>& rotation,
                                                            const std::array<double, 3>& point) noexcept {
            return {
                rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2],
                rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2],
                rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2]
            };
        }

        [[nodiscard]] bool contains(const cv::Rect2f& box, const cv::Point2f& point) noexcept {
            return point.x >= box.x &&
                   point.y >= box.y &&
                   point.x <= box.x + box.width &&
                   point.y <= box.y + box.height;
        }
    }

    bool LidarDetectionAssociator::initialize(const stereo::StereoCalibration& calibration,
                                              const core::SensorExtrinsics& extrinsics,
                                              std::string coordinate_frame) {
        initialized_ = false;

        if (!calibration.loaded() || coordinate_frame.empty()) return false;

        std::array<double, 9> body_from_camera{};
        std::array<double, 9> body_from_lidar{};

        if (!quaternionToMatrix(extrinsics.left_camera.rotation_xyzw, body_from_camera) ||
            !quaternionToMatrix(extrinsics.lidar.rotation_xyzw, body_from_lidar)) {
            return false;
        }

        /*
         * Existing Parallax extrinsics are consumed as body_from_sensor.
         *
         * p_body = R_body_lidar * p_lidar + t_body_lidar
         * p_raw  = R_body_camera^T * (p_body - t_body_camera)
         * p_rect = R1 * p_raw
         *
         * Pre-compose that chain once at startup.
         */
        const auto raw_from_body = transpose(body_from_camera);
        const auto raw_from_lidar = multiply(raw_from_body, body_from_lidar);
        rectified_from_lidar_rotation_ = multiply(calibration.R1(), raw_from_lidar);

        const std::array<double, 3> body_translation_delta{
            extrinsics.lidar.translation_m[0] - extrinsics.left_camera.translation_m[0],
            extrinsics.lidar.translation_m[1] - extrinsics.left_camera.translation_m[1],
            extrinsics.lidar.translation_m[2] - extrinsics.left_camera.translation_m[2]
        };

        const auto raw_translation = transformVector(raw_from_body, body_translation_delta);
        rectified_from_lidar_translation_m_ = transformVector(calibration.R1(), raw_translation);

        const auto& metadata = calibration.metadata();
        const auto& p1 = calibration.P1();

        width_ = metadata.image_width;
        height_ = metadata.image_height;
        fx_px_ = static_cast<float>(p1[0]);
        fy_px_ = static_cast<float>(p1[5]);
        cx_px_ = static_cast<float>(p1[2]);
        cy_px_ = static_cast<float>(p1[6]);
        coordinate_frame_ = std::move(coordinate_frame);

        owned_rectified_to_rgb_x_.clear();
        owned_rectified_to_rgb_y_.clear();
        rectified_to_rgb_x_ = &calibration.leftMapX();
        rectified_to_rgb_y_ = &calibration.leftMapY();

        const std::size_t pixels = static_cast<std::size_t>(width_) * height_;

        initialized_ =
            width_ > 0 &&
            height_ > 0 &&
            std::isfinite(fx_px_) && fx_px_ > 0.0F &&
            std::isfinite(fy_px_) && fy_px_ > 0.0F &&
            std::isfinite(cx_px_) &&
            std::isfinite(cy_px_) &&
            finiteArray(rectified_from_lidar_rotation_) &&
            finiteArray(rectified_from_lidar_translation_m_) &&
            rectified_to_rgb_x_ != nullptr &&
            rectified_to_rgb_y_ != nullptr &&
            rectified_to_rgb_x_->size() == pixels &&
            rectified_to_rgb_y_->size() == pixels;

        return initialized_;
    }

    bool LidarDetectionAssociator::initialize(
        std::uint32_t width,
        std::uint32_t height,
        float fx_px,
        float fy_px,
        float cx_px,
        float cy_px,
        std::array<double, 9> rectified_from_lidar_rotation,
        std::array<double, 3> rectified_from_lidar_translation_m,
        std::vector<float> rectified_to_rgb_x,
        std::vector<float> rectified_to_rgb_y,
        std::string coordinate_frame) {

        initialized_ = false;

        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        if (width == 0 ||
            height == 0 ||
            !std::isfinite(fx_px) || fx_px <= 0.0F ||
            !std::isfinite(fy_px) || fy_px <= 0.0F ||
            !std::isfinite(cx_px) ||
            !std::isfinite(cy_px) ||
            !finiteArray(rectified_from_lidar_rotation) ||
            !finiteArray(rectified_from_lidar_translation_m) ||
            rectified_to_rgb_x.size() != pixels ||
            rectified_to_rgb_y.size() != pixels ||
            coordinate_frame.empty()) {
            return false;
        }

        width_ = width;
        height_ = height;
        fx_px_ = fx_px;
        fy_px_ = fy_px;
        cx_px_ = cx_px;
        cy_px_ = cy_px;

        rectified_from_lidar_rotation_ = std::move(rectified_from_lidar_rotation);
        rectified_from_lidar_translation_m_ = std::move(rectified_from_lidar_translation_m);

        owned_rectified_to_rgb_x_ = std::move(rectified_to_rgb_x);
        owned_rectified_to_rgb_y_ = std::move(rectified_to_rgb_y);
        rectified_to_rgb_x_ = &owned_rectified_to_rgb_x_;
        rectified_to_rgb_y_ = &owned_rectified_to_rgb_y_;

        coordinate_frame_ = std::move(coordinate_frame);
        initialized_ = true;
        return true;
    }

    bool LidarDetectionAssociator::project(const lidar::LidarPoint& point,
                                           ProjectedHit& hit) const noexcept {
        if (!initialized_ ||
            !point.valid ||
            !std::isfinite(point.range_m) ||
            point.range_m <= 0.0F ||
            !std::isfinite(point.angle_rad) ||
            rectified_to_rgb_x_ == nullptr ||
            rectified_to_rgb_y_ == nullptr) {
            return false;
        }

        /*
         * Keep the SLAMTEC scan angle in the sensor's native measurement
         * convention here. The sign reversal in Publisher::publishLidarScan()
         * is a Foxglove presentation correction only; it must not leak back
         * into camera/LiDAR calibration or physical association math.
         */
        const double theta = static_cast<double>(point.angle_rad);
        const std::array<double, 3> lidar_point{
            static_cast<double>(point.range_m) * std::cos(theta),
            static_cast<double>(point.range_m) * std::sin(theta),
            0.0
        };

        const auto rotated = transformVector(rectified_from_lidar_rotation_, lidar_point);
        const std::array<double, 3> rectified{
            rotated[0] + rectified_from_lidar_translation_m_[0],
            rotated[1] + rectified_from_lidar_translation_m_[1],
            rotated[2] + rectified_from_lidar_translation_m_[2]
        };

        if (!finiteArray(rectified) || rectified[2] <= 1.0e-4) return false;

        const double rectified_x = static_cast<double>(fx_px_) * rectified[0] / rectified[2] + cx_px_;
        const double rectified_y = static_cast<double>(fy_px_) * rectified[1] / rectified[2] + cy_px_;

        if (!std::isfinite(rectified_x) || !std::isfinite(rectified_y)) return false;

        const auto x = static_cast<long>(std::lround(rectified_x));
        const auto y = static_cast<long>(std::lround(rectified_y));

        if (x < 0 ||
            y < 0 ||
            x >= static_cast<long>(width_) ||
            y >= static_cast<long>(height_)) {
            return false;
        }

        const std::size_t index =
            static_cast<std::size_t>(y) * width_ + static_cast<std::size_t>(x);

        const float source_x = (*rectified_to_rgb_x_)[index];
        const float source_y = (*rectified_to_rgb_y_)[index];

        if (!std::isfinite(source_x) ||
            !std::isfinite(source_y) ||
            source_x < 0.0F ||
            source_y < 0.0F ||
            source_x >= static_cast<float>(width_) ||
            source_y >= static_cast<float>(height_)) {
            return false;
        }

        hit.position_m = {
            static_cast<float>(rectified[0]),
            static_cast<float>(rectified[1]),
            static_cast<float>(rectified[2])
        };
        hit.source_pixel = {source_x, source_y};
        hit.range_m = point.range_m;
        return true;
    }

    bool LidarDetectionAssociator::associate(
        const DetectionSet& detections,
        const core::ProductMetadata& semantic_metadata,
        const core::Product<lidar::LidarScan>& scan,
        Object3DSet& output) const {

        output = {};
        output.query = detections.query;
        output.query_revision = detections.query_revision;

        if (!initialized_ ||
            !detections.valid() ||
            !semantic_metadata.valid ||
            !semantic_metadata.observation.valid() ||
            !scan.valid() ||
            !scan.payload ||
            !scan.payload->valid() ||
            scan.metadata.observation.source != core::SourceId::Rplidar) {
            return false;
        }

        if (detections.empty()) return true;

        const auto source_delta =
            scan.metadata.timestamp >= semantic_metadata.timestamp
                ? scan.metadata.timestamp - semantic_metadata.timestamp
                : semantic_metadata.timestamp - scan.metadata.timestamp;

        const auto association_timestamp = std::chrono::steady_clock::now();

        for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
            const auto& box = detections.boxes[detection_index];

            if (!std::isfinite(box.x) ||
                !std::isfinite(box.y) ||
                !std::isfinite(box.width) ||
                !std::isfinite(box.height) ||
                box.width <= 0.0F ||
                box.height <= 0.0F) {
                continue;
            }

            const cv::Point2f center{
                box.x + box.width * 0.5F,
                box.y + box.height * 0.5F
            };

            ProjectedHit best_hit{};
            float best_score = std::numeric_limits<float>::infinity();
            bool found = false;

            for (const auto& point : scan.payload->points) {
                ProjectedHit candidate{};
                if (!project(point, candidate) || !contains(box, candidate.source_pixel)) {
                    continue;
                }

                const float half_width = std::max(box.width * 0.5F, 1.0F);
                const float half_height = std::max(box.height * 0.5F, 1.0F);

                const float dx = (candidate.source_pixel.x - center.x) / half_width;
                const float dy = (candidate.source_pixel.y - center.y) / half_height;
                const float score = dx * dx + dy * dy;

                if (!found ||
                    score < best_score ||
                    (std::abs(score - best_score) < 1.0e-6F &&
                     candidate.range_m < best_hit.range_m)) {
                    best_hit = candidate;
                    best_score = score;
                    found = true;
                }
            }

            if (!found) continue;

            Object3D object{};
            object.label = detections.query;
            object.query_revision = detections.query_revision;
            object.semantic_confidence = detections.scores[detection_index];
            object.semantic_index = static_cast<std::uint32_t>(detection_index);
            object.image_box = box;
            object.image_space = detections.image_space;

            object.position_m = best_hit.position_m;

            // depth_m remains camera optical-axis depth. range_m is the direct
            // RPLIDAR line measurement that should be shown to the user.
            object.depth_m = best_hit.position_m[2];
            object.range_m = best_hit.range_m;
            object.coordinate_frame = coordinate_frame_;

            object.geometry = Object3DGeometry::Point;
            object.method = Object3DMethod::LidarAssociation;

            object.semantic_observation = semantic_metadata.observation;
            object.metric_observation = scan.metadata.observation;
            object.association_timestamp = association_timestamp;
            object.source_time_delta = source_delta;

            // Association quality here is image-space center proximity, not a
            // replacement for the detector confidence or SLAMTEC quality byte.
            object.support_quality = 1.0F / (1.0F + best_score);

            Object3DMetricEvidence lidar_evidence{};
            lidar_evidence.observation = scan.metadata.observation;
            lidar_evidence.position_m = object.position_m;
            lidar_evidence.depth_m = object.depth_m;
            lidar_evidence.range_m = object.range_m;
            lidar_evidence.source_time_delta = source_delta;
            lidar_evidence.support_quality = object.support_quality;
            object.lidar_evidence = lidar_evidence;

            output.objects.push_back(std::move(object));
        }

        return true;
    }

}

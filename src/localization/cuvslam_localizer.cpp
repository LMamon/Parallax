#include <parallax/localization/cuvslam_localizer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace parallax::localization {
    namespace {

        constexpr double MillimetersPerMeter = 1000.0;
        constexpr double Epsilon = 1.0e-9;
        constexpr double CalibrationToleranceMm = 1.0e-3;

        /**
         * Small internal rigid-transform representation.
         *
         * R and t represent parent_from_child:
         *
         *     p_parent = R * p_child + t
         *
         * That matches both cuVSLAM Camera::rig_from_camera and the semantics
         * already used when Parallax publishes SensorExtrinsics to Foxglove.
         */
        struct Transform {
            std::array<double, 9> rotation{1.0, 0.0, 0.0,
                                           0.0, 1.0, 0.0,
                                           0.0, 0.0, 1.0};

            std::array<double, 3> translation{0.0, 0.0, 0.0};
        };

        [[nodiscard]] bool finite(double value) noexcept { return std::isfinite(value); }

        [[nodiscard]] std::array<double, 9> quaternionToMatrix(const std::array<double, 4>& quaternion_xyzw) {
            double x = quaternion_xyzw[0];
            double y = quaternion_xyzw[1];
            double z = quaternion_xyzw[2];
            double w = quaternion_xyzw[3];

            const double norm = std::sqrt(x * x + y * y + z * z + w * w);
            if (!finite(norm) || norm <= Epsilon) {
                throw std::invalid_argument("sensor extrinsics contain an invalid quaternion");
            }

            // Normalize here so small calibration/config rounding errors do not
            // leak scaling into the rigid transform.
            x /= norm;
            y /= norm;
            z /= norm;
            w /= norm;

            return {1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),

                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),

                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y)};
        }

        [[nodiscard]] std::array<double, 9> transpose(const std::array<double, 9>& matrix) {
            return {matrix[0], matrix[3], matrix[6],
                    matrix[1], matrix[4], matrix[7],
                    matrix[2], matrix[5], matrix[8]};
        }

        [[nodiscard]] std::array<double, 9> multiply(const std::array<double, 9>& lhs, const std::array<double, 9>& rhs) {
            std::array<double, 9> output{};

            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t column = 0; column < 3; ++column) {
                    output[row * 3 + column] = lhs[row * 3 + 0] * rhs[0 * 3 + column] +
                                               lhs[row * 3 + 1] * rhs[1 * 3 + column] +
                                               lhs[row * 3 + 2] * rhs[2 * 3 + column];
                }
            }
            return output;
        }

        [[nodiscard]] std::array<double, 3> rotate(const std::array<double, 9>& rotation, const std::array<double, 3>& vector) {
            return {rotation[0] * vector[0] + rotation[1] * vector[1] + rotation[2] * vector[2],
                    rotation[3] * vector[0] + rotation[4] * vector[1] + rotation[5] * vector[2],
                    rotation[6] * vector[0] + rotation[7] * vector[1] + rotation[8] * vector[2]};
        }

        [[nodiscard]] std::array<float, 4> matrixToQuaternion(const std::array<double, 9>& rotation) {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            double w = 1.0;

            const double trace = rotation[0] + rotation[4] + rotation[8];

            if (trace > 0.0) {
                const double s = std::sqrt(trace + 1.0) * 2.0;

                w = 0.25 * s;
                x = (rotation[7] - rotation[5]) / s;
                y = (rotation[2] - rotation[6]) / s;
                z = (rotation[3] - rotation[1]) / s;

            } else if (rotation[0] > rotation[4] && rotation[0] > rotation[8]) {
                const double s = std::sqrt(1.0 + rotation[0] - rotation[4] - rotation[8]) * 2.0;

                w = (rotation[7] - rotation[5]) / s;
                x = 0.25 * s;
                y = (rotation[1] + rotation[3]) / s;
                z = (rotation[2] + rotation[6]) / s;

            } else if (rotation[4] > rotation[8]) {
                const double s = std::sqrt(1.0 + rotation[4] - rotation[0] - rotation[8]) * 2.0;

                w = (rotation[2] - rotation[6]) / s;
                x = (rotation[1] + rotation[3]) / s;
                y = 0.25 * s;
                z = (rotation[5] + rotation[7]) / s;

            } else {
                const double s = std::sqrt(1.0 + rotation[8] - rotation[0] - rotation[4]) * 2.0;

                w = (rotation[3] - rotation[1]) / s;
                x = (rotation[2] + rotation[6]) / s;
                y = (rotation[5] + rotation[7]) / s;
                z = 0.25 * s;
            }

            const double norm = std::sqrt(x * x + y * y + z * z + w * w);

            if (!finite(norm) || norm <= Epsilon) {
                throw std::invalid_argument("failed to convert rig rotation to quaternion");
            }

            return {static_cast<float>(x / norm),
                    static_cast<float>(y / norm),
                    static_cast<float>(z / norm),
                    static_cast<float>(w / norm)};
        }

        [[nodiscard]] cuvslam::Pose toCuVslamPose(const Transform& transform) {
            cuvslam::Pose pose{};

            pose.rotation = matrixToQuaternion(transform.rotation);

            pose.translation = {static_cast<float>(transform.translation[0]),
                                static_cast<float>(transform.translation[1]),
                                static_cast<float>(transform.translation[2])};

            return pose;
        }

        [[nodiscard]] Transform bodyFromRawLeftCamera(const core::RigidTransformConfig& extrinsics) {
            if (extrinsics.parent_frame.empty() || extrinsics.child_frame.empty()) {
                throw std::invalid_argument("left-camera sensor extrinsics require frame names");
            }

            for (double value : extrinsics.translation_m) {
                if (!finite(value)) {
                    throw std::invalid_argument("left-camera translation contains non-finite values");
                }
            }

            Transform transform{};
            transform.rotation = quaternionToMatrix(extrinsics.rotation_xyzw);
            transform.translation = extrinsics.translation_m;

            /*
             * SensorExtrinsics is already published as:
             *
             *     parent = stereo_body
             *     child  = camera_left_optical
             *
             * so its stored translation/quaternion describe the child camera
             * pose in the body frame. This is the same direction cuVSLAM calls
             * rig_from_camera; no inversion belongs here.
             */
            return transform;
        }

        [[nodiscard]] double rectifiedBaselineMeters(const stereo::StereoCalibration& calibration) {
            const auto& metadata = calibration.metadata();
            const auto& p1 = calibration.P1();
            const auto& p2 = calibration.P2();

            if (metadata.extrinsics_convention != "right_from_left") {
                throw std::invalid_argument("unsupported stereo extrinsics convention");
            }

            if (!finite(p1[0]) || !finite(p2[0]) || p1[0] <= 0.0 || p2[0] <= 0.0) {
                throw std::invalid_argument("rectified stereo focal length is invalid");
            }

            /*
             * OpenCV's rectified P2 is:
             *
             *     K [ I | Tx ]
             *
             * and the current calibration contains:
             *
             *     P2(0,3) = -fx * baseline
             *
             * The matrices were generated from millimeter calibration
             * extrinsics, so the translation recovered from P2 is millimeters.
             */
            const double baseline_mm = -p2[3] / p2[0];
            if (!finite(baseline_mm) || baseline_mm <= 0.0) {
                throw std::invalid_argument("rectified P2 does not encode a positive stereo baseline");
            }

            if (!finite(metadata.baseline_mm) ||
                metadata.baseline_mm <= 0.0 ||
                std::abs(baseline_mm - metadata.baseline_mm) > CalibrationToleranceMm) {

                throw std::invalid_argument("rectified P2 baseline disagrees with calibration metadata");
            }

            /*
             * Rectified stereo must have matching virtual intrinsics. cuVSLAM's
             * rectified_stereo_camera optimization assumes this canonical
             * horizontal pair rather than two arbitrary camera projections.
             */
            if (std::abs(p1[0] - p2[0]) > Epsilon ||
                std::abs(p1[5] - p2[5]) > Epsilon ||
                std::abs(p1[2] - p2[2]) > Epsilon ||
                std::abs(p1[6] - p2[6]) > Epsilon ||
                std::abs(p2[7]) > Epsilon ||
                std::abs(p2[11]) > Epsilon) {

                throw std::invalid_argument("calibration is not a canonical rectified stereo pair");
            }
            return baseline_mm / MillimetersPerMeter;
        }

        [[nodiscard]] cuvslam::Camera makeCamera(const std::array<double, 12>& projection,
                                                 std::uint32_t width,
                                                 std::uint32_t height,
                                                 const Transform& rig_from_camera) {
            cuvslam::Camera camera{};

            camera.size = {static_cast<std::int32_t>(width), static_cast<std::int32_t>(height)};
            camera.focal = {static_cast<float>(projection[0]), static_cast<float>(projection[5])};
            camera.principal = {static_cast<float>(projection[2]), static_cast<float>(projection[6])};

            /*
             * Rectification has already removed the original lens distortion.
             * Passing the raw camera distortion model here would apply geometry
             * that no longer matches the images submitted to Track().
             */
            camera.distortion.model = cuvslam::Distortion::Model::Pinhole;

            camera.distortion.parameters.clear();
            camera.rig_from_camera = toCuVslamPose(rig_from_camera);

            return camera;
        }
    }

    CuVslamLocalizer::~CuVslamLocalizer() { shutdown(); }

    cuvslam::Rig CuVslamLocalizer::makeRig(const stereo::StereoCalibration& calibration, const core::SensorExtrinsics& extrinsics) {
        if (!calibration.loaded()) {
            throw std::invalid_argument("cuVSLAM requires loaded stereo calibration");
        }

        const auto& metadata = calibration.metadata();
        if (metadata.image_width == 0 || metadata.image_height == 0) {
            throw std::invalid_argument("cuVSLAM calibration has invalid image dimensions");
        }

        /*
         * SensorExtrinsics gives us the physical/raw left optical camera pose
         * relative to stereo_body.
         */
        const Transform body_from_raw_left = bodyFromRawLeftCamera(extrinsics.left_camera);

        /*
         * OpenCV R1 rotates points from the original left optical frame into
         * the rectified virtual frame:
         *
         *     p_rect = R1 * p_raw
         *
         * cuVSLAM needs body_from_rectified_camera, so the camera-side change
         * of basis is R1^T:
         *
         *     body_from_rect = body_from_raw * raw_from_rect
         */
        Transform body_from_rectified_left{};
        body_from_rectified_left.rotation = multiply(body_from_raw_left.rotation, transpose(calibration.R1()));

        // Rectification rotates the virtual camera around the same optical
        // center, so the physical camera origin does not translate.
        body_from_rectified_left.translation = body_from_raw_left.translation;
        const double baseline_m = rectifiedBaselineMeters(calibration);

        /*
         * P2's negative projection translation means the right camera center is
         * +baseline along the rectified LEFT camera's x axis.
         *
         * Both virtual rectified cameras have the same orientation. Therefore
         * their body-frame rotations are identical and only their optical
         * centers differ.
         */
        const std::array<double, 3> right_center_from_left_center{baseline_m, 0.0, 0.0};
        const auto right_offset_in_body = rotate(body_from_rectified_left.rotation, right_center_from_left_center);

        Transform body_from_rectified_right = body_from_rectified_left;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            body_from_rectified_right.translation[axis] += right_offset_in_body[axis];
        }

        cuvslam::Rig rig{};
        rig.cameras.reserve(2);

        /*
         * Camera indices are part of the runtime contract:
         *
         *     0 -> rectified left
         *     1 -> rectified right
         *
         * CuVslamProducer must preserve this ordering when it later constructs
         * each ImageSet.
         */
        rig.cameras.push_back(makeCamera(calibration.P1(),
                                         metadata.image_width,
                                         metadata.image_height,
                                         body_from_rectified_left));

        rig.cameras.push_back(makeCamera(calibration.P2(),
                                         metadata.image_width,
                                         metadata.image_height,
                                         body_from_rectified_right));

        return rig;
    }

    bool CuVslamLocalizer::initialize(const stereo::StereoCalibration& calibration, const core::SensorExtrinsics& extrinsics) {
        shutdown();

        try {
            auto rig = makeRig(calibration, extrinsics);

            auto config = cuvslam::Odometry::GetDefaultConfig();

            // We have a synchronized overlapping stereo pair and no IMU in the
            // Phase 17 scope, so Multicamera is the correct visual mode.
            config.odometry_mode = cuvslam::Odometry::OdometryMode::Multicamera;
            config.multicam_mode = cuvslam::Odometry::MulticameraMode::Precision;

            config.use_gpu = true;

            // The submitted frames are already the P1/P2 rectified pair.
            config.rectified_stereo_camera = true;

            /*
             * cuVSLAM's SBA remains library-owned. Keeping async SBA enabled
             * does not make Parallax's Track() submission unordered; Track()
             * itself remains the synchronous state-machine boundary.
             */
            config.async_sba = true;

            // Force CUDA/cuBLAS/cuSolver initialization here so startup failure
            // happens during localization initialization, not on the first
            // demanded frame.
            cuvslam::WarmUpGPU();

            auto odometry = std::make_unique<cuvslam::Odometry>(rig, config);

            image_width_ = static_cast<std::int32_t>(calibration.metadata().image_width);
            image_height_ = static_cast<std::int32_t>(calibration.metadata().image_height);

            rig_ = std::move(rig);
            config_ = std::move(config);
            odometry_ = std::move(odometry);

            last_timestamp_ns_ = -1;

            return true;

        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize cuVSLAM: " << e.what() << '\n';
            shutdown();
            return false;
        }
    }

    CuVslamPoseEstimate CuVslamLocalizer::track(const CuVslamFrame& frame) {
        if (!odometry_) throw std::logic_error("cuVSLAM track called before initialization");

        if (!frame.left || !frame.right) {
            throw std::invalid_argument("cuVSLAM stereo frame contains null image pointers");
        }

        if (frame.width != image_width_ || frame.height != image_height_) {
            throw std::invalid_argument("cuVSLAM frame dimensions do not match calibration");
        }

        if (frame.timestamp_ns < 0) throw std::invalid_argument("cuVSLAM frame timestamp is invalid");

        /*
         * cuVSLAM is stateful and timestamps must increase. The producer will
         * later enforce ordered ProductStore consumption, while this check
         * protects the adapter itself from direct reordered submission.
         */
        if (last_timestamp_ns_ >= 0 && frame.timestamp_ns <= last_timestamp_ns_) {
            throw std::invalid_argument("cuVSLAM frame timestamps must be strictly increasing");
        }

        if (frame.gpu_memory && (frame.left_pitch < frame.width ||frame.right_pitch < frame.width)) {

            throw std::invalid_argument("cuVSLAM GPU image pitch is smaller than image width");
        }

        cuvslam::Image left{};

        left.pixels = frame.left;
        left.width = frame.width;
        left.height = frame.height;
        left.pitch = frame.left_pitch;
        left.encoding = cuvslam::ImageData::Encoding::MONO;
        left.data_type = cuvslam::ImageData::DataType::UINT8;
        left.is_gpu_mem = frame.gpu_memory;
        left.timestamp_ns = frame.timestamp_ns;
        left.camera_index = 0;

        cuvslam::Image right{};

        right.pixels = frame.right;
        right.width = frame.width;
        right.height = frame.height;
        right.pitch = frame.right_pitch;
        right.encoding = cuvslam::ImageData::Encoding::MONO;
        right.data_type = cuvslam::ImageData::DataType::UINT8;
        right.is_gpu_mem = frame.gpu_memory;
        right.timestamp_ns = frame.timestamp_ns;
        right.camera_index = 1;

        std::vector<cuvslam::Image> images;
        images.reserve(2);
        images.push_back(left);
        images.push_back(right);

        /*
         * Track() is intentionally called synchronously here. We do not insert
         * a CUDA-wide synchronization around it; the future producer will add
         * only the generation-specific readiness dependency needed before these
         * image pointers are submitted.
         */
        const cuvslam::PoseEstimate estimate = odometry_->Track(images);

        /*
         * Even a tracking-loss result consumed this timestamp. Advancing the
         * cursor prevents a failed pose from making a later frame look like a
         * legal retry of the same estimator time.
         */
        last_timestamp_ns_ = frame.timestamp_ns;

        CuVslamPoseEstimate result{};
        result.timestamp_ns = estimate.timestamp_ns;
        result.world_from_rig = estimate.world_from_rig;

        return result;
    }

    bool CuVslamLocalizer::reset() {
        if (!rig_ || !config_) return false;

        try {
            /*
             * Construct the replacement before discarding the old tracker. A
             * failed reset therefore does not leave us pretending a new world
             * epoch exists when construction actually failed.
             */
            auto replacement = std::make_unique<cuvslam::Odometry>(*rig_, *config_);

            odometry_ = std::move(replacement);
            last_timestamp_ns_ = -1;

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to reset cuVSLAM: " << e.what() << '\n';
            return false;
        }
    }

    void CuVslamLocalizer::shutdown() noexcept {
        // Destroy the stateful tracker first; its destructor may release
        // internal CUDA/cuVSLAM resources that were configured from rig_.
        odometry_.reset();

        rig_.reset();
        config_.reset();

        image_width_ = 0;
        image_height_ = 0;
        last_timestamp_ns_ = -1;
    }
}
#pragma once

#include <parallax/core/product.hpp>
#include <parallax/core/sensor_extrinsics.hpp>
#include <parallax/lidar/frame_types.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/perception/object3d.hpp>
#include <parallax/stereo/calibration.hpp>

#include <opencv2/core/types.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace parallax::perception {

    /**
     * Associates an image-space semantic detection with a direct RPLIDAR hit.
     *
     * LiDAR is projected into the same RgbLeft image space used by NanoOWL.
     * A projected return must land inside the detection box. When several
     * returns land inside one box, the return closest to the box center wins.
     *
     * The selected 3D point is expressed in the P1 rectified-left camera frame
     * so downstream localization uses the same transform path as stereo
     * Object3D observations.
     */
    class LidarDetectionAssociator {
        public:
            LidarDetectionAssociator() = default;

            bool initialize(const stereo::StereoCalibration& calibration,
                            const core::SensorExtrinsics& extrinsics,
                            std::string coordinate_frame);

            // Explicit initialization keeps projection/association unit-testable
            // without loading calibration files.
            bool initialize(std::uint32_t width,
                            std::uint32_t height,
                            float fx_px,
                            float fy_px,
                            float cx_px,
                            float cy_px,
                            std::array<double, 9> rectified_from_lidar_rotation,
                            std::array<double, 3> rectified_from_lidar_translation_m,
                            std::vector<float> rectified_to_rgb_x,
                            std::vector<float> rectified_to_rgb_y,
                            std::string coordinate_frame);

            [[nodiscard]] bool initialized() const noexcept { return initialized_; }

            bool associate(const DetectionSet& detections,
                           const core::ProductMetadata& semantic_metadata,
                           const core::Product<lidar::LidarScan>& scan,
                           Object3DSet& output) const;

        private:
            struct ProjectedHit {
                std::array<float, 3> position_m{};
                cv::Point2f source_pixel{};
                float range_m = 0.0F;
            };

            [[nodiscard]] bool project(const lidar::LidarPoint& point,
                                       ProjectedHit& hit) const noexcept;

            std::uint32_t width_ = 0;
            std::uint32_t height_ = 0;

            float fx_px_ = 0.0F;
            float fy_px_ = 0.0F;
            float cx_px_ = 0.0F;
            float cy_px_ = 0.0F;

            std::array<double, 9> rectified_from_lidar_rotation_{};
            std::array<double, 3> rectified_from_lidar_translation_m_{};

            // Production uses calibration-owned maps without duplicating them.
            // Explicit test initialization owns its supplied maps here.
            const std::vector<float>* rectified_to_rgb_x_ = nullptr;
            const std::vector<float>* rectified_to_rgb_y_ = nullptr;
            std::vector<float> owned_rectified_to_rgb_x_;
            std::vector<float> owned_rectified_to_rgb_y_;

            std::string coordinate_frame_;
            bool initialized_ = false;
    };

}

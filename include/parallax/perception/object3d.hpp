#pragma once

#include <parallax/core/product.hpp>
#include <parallax/perception/image_space.hpp>

#include <opencv2/core/types.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace parallax::perception {

    enum class Object3DGeometry : std::uint8_t {
        Unknown = 0,
        Point,
        ImageSupportedGeometry,
        Surface,
        ObservedExtent,
        PhysicalExtent
    };

    enum class Object3DMethod : std::uint8_t {
        Unknown = 0,
        StereoRoi,
        StereoMask,
        LidarAssociation,
        StereoLidarRefined
    };

    struct Object3DMetricEvidence {
        core::SourceObservation observation{};
        std::array<float, 3> position_m{};
        float depth_m = 0.0F;

        // Direct sensor-line range is populated by LiDAR. Stereo depth is
        // optical-axis Z, so stereo evidence intentionally leaves this zero.
        float range_m = 0.0F;

        std::chrono::steady_clock::duration source_time_delta{};
        float support_quality = 0.0F;

        // TODO: split object3D into explicit components. Represent sensor/method-specific 
        // states with typed variants rather than encoding valid states through fields + valid().
        [[nodiscard]] bool valid() const noexcept {
            if (!observation.valid() ||
                !std::isfinite(position_m[0]) ||
                !std::isfinite(position_m[1]) ||
                !std::isfinite(position_m[2]) ||
                !std::isfinite(depth_m) ||
                depth_m <= 0.0F ||
                !std::isfinite(support_quality) ||
                support_quality < 0.0F ||
                support_quality > 1.0F) {

                return false;
            }

            if (observation.source == core::SourceId::Rplidar) {
                return std::isfinite(range_m) && range_m > 0.0F;
            }

            return observation.source == core::SourceId::StereoCamera;
        }
    };

    struct Object3D {
        std::string label;
        std::uint64_t query_revision = 0;
        float semantic_confidence = 0.0F;
        
        std::uint32_t semantic_index = 0;
        std::vector<std::array<float, 3>> surface_points_m;

        /*
         * Axis-aligned bounds of stereo-supported surface measurements in
         * coordinate_frame. This is observed extent, not a claim about hidden
         * or back-side physical dimensions.
         */
        std::array<float, 3> observed_extent_center_m{};
        std::array<float, 3> observed_extent_size_m{};
        std::uint32_t observed_extent_support = 0;
        
        cv::Rect2f image_box{};
        ImageSpace image_space = ImageSpace::Unknown;

        cv::Rect2f depth_roi{};
        ImageSpace depth_image_space = ImageSpace::Unknown;

        // Image-supported rectangle projected at the representative depth.
        // Valid when geometry == ImageSupportedGeometry.
        std::array<std::array<float, 3>, 4> image_supported_corners_m{};

        std::array<float, 3> position_m{};
        float depth_m = 0.0F;

        // Direct sensor-line range when available. For LidarAssociation this is
        // the selected RPLIDAR return; depth_m remains camera optical-axis depth.
        float range_m = 0.0F;
        std::string coordinate_frame;

        Object3DGeometry geometry = Object3DGeometry::Unknown;
        Object3DMethod method = Object3DMethod::Unknown;

        // Association combines semantic and metric observations, so both
        // provenances remain explicit instead of replacing one with the other.
        core::SourceObservation semantic_observation{};
        core::SourceObservation metric_observation{};

        std::chrono::steady_clock::time_point association_timestamp{};
        std::chrono::steady_clock::duration source_time_delta{};

        // Non-zero only when the source is a persistent Phase 14 target.
        std::uint64_t track_id = 0;

        float support_quality = 0.0F;

        // Preserve independent metric evidence even when policy selects one
        // sensor as the representative Object3D position. This lets downstream
        // consumers inspect stereo and LiDAR agreement without reconstructing
        // it from discarded intermediate products.
        std::optional<Object3DMetricEvidence> stereo_evidence;
        std::optional<Object3DMetricEvidence> lidar_evidence;

        [[nodiscard]] bool valid() const noexcept {
            // TODO: split object3D into explicit components. Represent sensor/method-specific 
            // states with typed variants rather than encoding valid states through fields + valid().
            return !label.empty() &&
                   query_revision != 0 &&
                   image_space != ImageSpace::Unknown &&
                   semantic_observation.valid() &&
                   metric_observation.valid() &&
                   !coordinate_frame.empty() &&
                   geometry != Object3DGeometry::Unknown &&
                   method != Object3DMethod::Unknown &&
                   std::isfinite(depth_m) &&
                   depth_m > 0.0F &&
                   ((method != Object3DMethod::LidarAssociation && method != Object3DMethod::StereoLidarRefined) ||
                    (std::isfinite(range_m) && range_m > 0.0F)) &&
                   (!stereo_evidence || (stereo_evidence->valid() &&
                     stereo_evidence->observation.source == core::SourceId::StereoCamera)) &&
                   (!lidar_evidence || (lidar_evidence->valid() &&
                     lidar_evidence->observation.source == core::SourceId::Rplidar)) &&
                   (method != Object3DMethod::StereoLidarRefined || (stereo_evidence.has_value() && lidar_evidence.has_value())) &&
                   std::isfinite(position_m[0]) &&
                   std::isfinite(position_m[1]) &&
                   std::isfinite(position_m[2]) &&
                   (geometry != Object3DGeometry::ObservedExtent ||
                    (observed_extent_support > 0 &&
                        std::isfinite(observed_extent_center_m[0]) &&
                        std::isfinite(observed_extent_center_m[1]) &&
                        std::isfinite(observed_extent_center_m[2]) &&
                        std::isfinite(observed_extent_size_m[0]) &&
                        std::isfinite(observed_extent_size_m[1]) &&
                        std::isfinite(observed_extent_size_m[2]) &&
                        observed_extent_size_m[0] > 0.0F &&
                        observed_extent_size_m[1] > 0.0F &&
                        observed_extent_size_m[2] > 0.0F));
        }

        [[nodiscard]] bool persistent() const noexcept { return track_id != 0; }
    };

    struct Object3DSet {
        std::string query;
        std::uint64_t query_revision = 0;
        std::vector<Object3D> objects;

        [[nodiscard]] bool valid() const noexcept {
            if (query.empty() || query_revision == 0) return false;

            for (const auto& object : objects) {
                if (!object.valid()) return false;
            }

            return true;
        }

        [[nodiscard]] std::size_t size() const noexcept { return objects.size(); }
        [[nodiscard]] bool empty() const noexcept { return objects.empty(); }
    };
}
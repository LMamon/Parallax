#pragma once

#include <parallax/core/product.hpp>
#include <parallax/perception/image_space.hpp>

#include <opencv2/core/types.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace parallax::perception {

    enum class Object3DGeometry : std::uint8_t {
        Unknown = 0,
        Point,
        ImageSupportedGeometry,
        Surface,
        PhysicalExtent
    };

    enum class Object3DMethod : std::uint8_t {
        Unknown = 0,
        StereoRoi,
        StereoMask,
        LidarAssociation,
        StereoLidarRefined
    };

    struct Object3D {
        std::string label;
        std::uint64_t query_revision = 0;
        float semantic_confidence = 0.0F;
        
        std::uint32_t semantic_index = 0;
        std::vector<std::array<float, 3>> surface_points_m;
        
        cv::Rect2f image_box{};
        ImageSpace image_space = ImageSpace::Unknown;

        cv::Rect2f depth_roi{};
        ImageSpace depth_image_space = ImageSpace::Unknown;

        // Image-supported rectangle projected at the representative depth.
        // Valid when geometry == ImageSupportedGeometry.
        std::array<std::array<float, 3>, 4> image_supported_corners_m{};

        std::array<float, 3> position_m{};
        float depth_m = 0.0F;
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

        [[nodiscard]] bool valid() const noexcept {
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
                   std::isfinite(position_m[0]) &&
                   std::isfinite(position_m[1]) &&
                   std::isfinite(position_m[2]);
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
#pragma once

#include <parallax/core/product.hpp>
#include <parallax/perception/object3d.hpp>

#include <cstdint>
#include <string>

namespace parallax::perception {

    /*
     * Metric result of intersecting a semantic mask with stereo depth.
     *
     * This is not a second full-resolution depth allocation. The mask/depth
     * intersection is represented by bounded XYZ support in Object3D, which is
     * enough for scene visualization, observed extent, and tracker correction.
     * The original SegmentationMask and DepthFrame remain independently
     * available when a consumer needs their image-domain representations.
     */
    struct SegmentedDepth {
        std::string query;
        std::uint64_t query_revision = 0;
        core::SourceObservation source_observation{};
        Object3D object{};

        [[nodiscard]] bool valid() const noexcept {
            // TODO: SPLIT INTO EXPLICIT COMPOENENTS
            return !query.empty() &&
                   query_revision != 0 &&
                   source_observation.valid() &&
                   object.valid() &&
                   object.method == Object3DMethod::StereoMask &&
                   object.semantic_observation == source_observation &&
                   object.metric_observation == source_observation;
        }

        [[nodiscard]] Object3DSet asObject3DSet() const {
            Object3DSet set{};
            set.query = query;
            set.query_revision = query_revision;
            set.objects.push_back(object);
            return set;
        }
    };
}

#include <parallax/mapping/spatial_mesh_producer.hpp>
#include <parallax/mapping/spatial_mesh_snapshot.hpp>
#include <parallax/mapping/spatial_mesh_state.hpp>
#include <parallax/core/execution_context.hpp>
#include <parallax/isp/frame_types.hpp>

#include <memory>

namespace parallax::mapping {

SpatialMeshProducer::SpatialMeshProducer(
    const MappingConfig& config,
    const parallax::stereo::StereoCalibration& calibration,
    SpatialMap& map,
    parallax::core::ProductStore& products)
    : config_(config), map_(map), products_(products) {
    const auto& p1 = calibration.P1();
    camera_ = nvblox::Camera(static_cast<float>(p1[0]), static_cast<float>(p1[5]),
                             static_cast<float>(p1[2]), static_cast<float>(p1[6]),
                             static_cast<int>(calibration.metadata().image_width),
                             static_cast<int>(calibration.metadata().image_height));
}

std::string_view SpatialMeshProducer::name() const noexcept {
    return "mapping.spatial_mesh";
}

const std::vector<parallax::core::ProductId>& SpatialMeshProducer::inputs() const noexcept {
    return inputs_;
}

const std::vector<parallax::core::ProductId>& SpatialMeshProducer::outputs() const noexcept {
    return outputs_;
}

const std::vector<parallax::core::CompatibleInputRequirement>&
SpatialMeshProducer::compatible_inputs() const noexcept {
    return compatible_inputs_;
}

parallax::core::ExecutionPolicy SpatialMeshProducer::execution_policy() const noexcept {
    parallax::core::ExecutionPolicy policy{};
    policy.target_hz = config_.mesh_update_rate_hz;
    policy.drop_policy = parallax::core::DropPolicy::Supersede;
    policy.priority = 2;
    policy.affinity = parallax::core::ResourceAffinity::Gpu;
    policy.stateful = false;
    return policy;
}

parallax::core::SubmitResult SpatialMeshProducer::submit(
    parallax::core::ExecutionContext& context) {
    const auto map_state = products_.latest<SpatialMapState>(
        parallax::core::ProductId::SpatialMapState);
    if (!map_state || !map_state->valid() || !map_state->payload || !map_.initialized()) {
        return parallax::core::SubmitResult::NoWork;
    }
    if (last_revision_ && *last_revision_ == map_state->payload->revision) {
        return parallax::core::SubmitResult::NoWork;
    }
    if (map_.epoch() != map_state->payload->localization_epoch) {
        return parallax::core::SubmitResult::NoWork;
    }

    // Phase A preserves the existing color-mesh behavior but moves all appearance
    // and mesh materialization work out of the persistent integrator.
    if (!config_.color_enabled) {
        return parallax::core::SubmitResult::NoWork;
    }

    const auto rgb = products_.find_observation<parallax::isp::RectifiedStereoFrame>(
        parallax::core::ProductId::RectifiedRgb, map_state->metadata.observation);
    if (!rgb || !rgb->valid() || !rgb->payload || !rgb->payload->left.isAllocated()) {
        return parallax::core::SubmitResult::NoWork;
    }
    if (!context.waitFor(rgb->completion, map_.cudaStream())) {
        return parallax::core::SubmitResult::Failed;
    }

    const auto& cf = *rgb->payload;
    nvblox::ColorImageConstView color_view(
        static_cast<int>(cf.height), static_cast<int>(cf.width),
        static_cast<int>(cf.left.pitch()), 3,
        reinterpret_cast<const nvblox::Color*>(cf.left.dataAs<std::uint8_t>()));
    const nvblox::MaskedColorImageConstView masked_color(
        color_view, nvblox::kMaskActiveEverywhere);

    nvblox::Transform world_from_camera = nvblox::Transform::Identity();
    const auto& r = map_state->payload->world_from_camera_rotation;
    world_from_camera.linear() <<
        r[0], r[1], r[2],
        r[3], r[4], r[5],
        r[6], r[7], r[8];
    const auto& t = map_state->payload->world_from_camera_translation_m;
    world_from_camera.translation() = nvblox::Vector3f(t[0], t[1], t[2]);

    map_.mapper().integrateColor(masked_color, world_from_camera, camera_);
    map_.mapper().updateFlatColorMesh();

    auto state = std::make_shared<SpatialMeshState>();
    state->localization_epoch = map_state->payload->localization_epoch;
    state->integrated_frames = map_state->payload->integrated_frames;
    state->mesh_revision = ++mesh_revision_;

    if (!buildSpatialMeshSnapshot(map_.mapper().flat_color_mesh(), state.get())) {
        return parallax::core::SubmitResult::NoWork;
    }

    auto metadata = map_state->metadata;
    metadata.production_timestamp = parallax::core::ExecutionContext::now();
    metadata.valid = true;
    std::shared_ptr<const SpatialMeshState> published = std::move(state);
    products_.publish(parallax::core::make_product(
        parallax::core::ProductId::SpatialMesh,
        metadata,
        std::move(published)));
    last_revision_ = map_state->payload->revision;
    return parallax::core::SubmitResult::Submitted;
}

}  // namespace parallax::mapping

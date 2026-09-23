#include <parallax/mapping/tsdf_snapshot_producer.hpp>
#include <parallax/mapping/spatial_tsdf_state.hpp>
#include <parallax/mapping/tsdf_snapshot.hpp>
#include <parallax/core/execution_context.hpp>

#include <memory>

namespace parallax::mapping {

TsdfSnapshotProducer::TsdfSnapshotProducer(
    const MappingConfig& config,
    SpatialMap& map,
    parallax::core::ProductStore& products)
    : config_(config), map_(map), products_(products) {}

std::string_view TsdfSnapshotProducer::name() const noexcept {
    return "mapping.tsdf_snapshot";
}

const std::vector<parallax::core::ProductId>& TsdfSnapshotProducer::inputs() const noexcept {
    return inputs_;
}

const std::vector<parallax::core::ProductId>& TsdfSnapshotProducer::outputs() const noexcept {
    return outputs_;
}

parallax::core::ExecutionPolicy TsdfSnapshotProducer::execution_policy() const noexcept {
    parallax::core::ExecutionPolicy policy{};
    policy.target_hz = config_.tsdf_visualization_rate_hz;
    policy.drop_policy = parallax::core::DropPolicy::Supersede;
    policy.priority = 2;
    policy.affinity = parallax::core::ResourceAffinity::Gpu;
    policy.stateful = false;
    return policy;
}

parallax::core::SubmitResult TsdfSnapshotProducer::submit(
    parallax::core::ExecutionContext&) {
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

    auto state = std::make_shared<SpatialTsdfState>();
    state->localization_epoch = map_state->payload->localization_epoch;
    state->integrated_frames = map_state->payload->integrated_frames;
    state->epoch_resets = map_state->payload->epoch_resets;
    state->allocated_blocks = map_state->payload->allocated_blocks;

    const auto& c = map_state->payload->world_from_camera_translation_m;
    const nvblox::Vector3f center(c[0], c[1], c[2]);
    const TsdfSnapshotConfig snapshot_config{
        config_.voxel_size_m,
        config_.debug_columns,
        config_.debug_rows,
        config_.debug_slices,
        config_.debug_surface_band_m};

    if (!buildTsdfSnapshot(map_.mapper().tsdf_layer(),
                           center,
                           map_.stream(),
                           snapshot_config,
                           state.get())) {
        return parallax::core::SubmitResult::Failed;
    }

    auto metadata = map_state->metadata;
    metadata.production_timestamp = parallax::core::ExecutionContext::now();
    metadata.valid = true;
    std::shared_ptr<const SpatialTsdfState> published = std::move(state);
    products_.publish(parallax::core::make_product(
        parallax::core::ProductId::SpatialTsdf,
        metadata,
        std::move(published)));
    last_revision_ = map_state->payload->revision;
    return parallax::core::SubmitResult::Submitted;
}

}  // namespace parallax::mapping

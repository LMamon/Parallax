#include <parallax/mapping/esdf_producer.hpp>

#include <parallax/core/execution_context.hpp>
#include <parallax/mapping/spatial_esdf_state.hpp>

#include <memory>

namespace parallax::mapping {

EsdfProducer::EsdfProducer(
    const MappingConfig& config,
    SpatialMap& map,
    parallax::core::ProductStore& products)
    : config_(config), map_(map), products_(products) {}

std::string_view EsdfProducer::name() const noexcept {
    return "mapping.esdf";
}

const std::vector<parallax::core::ProductId>&
EsdfProducer::inputs() const noexcept {
    return inputs_;
}

const std::vector<parallax::core::ProductId>&
EsdfProducer::outputs() const noexcept {
    return outputs_;
}

parallax::core::ExecutionPolicy
EsdfProducer::execution_policy() const noexcept {
    parallax::core::ExecutionPolicy policy{};
    policy.target_hz = config_.integration_rate_hz;
    policy.drop_policy = parallax::core::DropPolicy::Supersede;
    policy.priority = 4;
    policy.affinity = parallax::core::ResourceAffinity::Gpu;
    policy.stateful = true;
    return policy;
}

parallax::core::SubmitResult EsdfProducer::submit(
    parallax::core::ExecutionContext&) {

    const auto map_state = products_.latest<SpatialMapState>(
        parallax::core::ProductId::SpatialMapState);

    if (!map_state ||
        !map_state->valid() ||
        !map_state->payload ||
        !map_.initialized()) {
        return parallax::core::SubmitResult::NoWork;
    }

    const auto& state = *map_state->payload;

    if (map_.epoch() != state.localization_epoch) {
        return parallax::core::SubmitResult::NoWork;
    }

    if (last_map_revision_ &&
        *last_map_revision_ == state.revision) {
        return parallax::core::SubmitResult::NoWork;
    }

    if (epoch_ != state.localization_epoch) {
        epoch_ = state.localization_epoch;
        esdf_revision_ = 0;
        last_map_revision_.reset();
    }

    // Incrementally update only ESDF blocks affected by TSDF changes.
    map_.mapper().updateEsdf();

    ++esdf_revision_;

    auto esdf_state = std::make_shared<SpatialEsdfState>();
    esdf_state->localization_epoch = state.localization_epoch;
    esdf_state->map_revision = state.revision;
    esdf_state->esdf_revision = esdf_revision_;
    esdf_state->allocated_blocks =
        map_.mapper().esdf_layer().numAllocatedBlocks();
    esdf_state->voxel_size_m = config_.voxel_size_m;

    auto metadata = map_state->metadata;
    metadata.production_timestamp =
        parallax::core::ExecutionContext::now();
    metadata.valid = true;

    std::shared_ptr<const SpatialEsdfState> published =
        std::move(esdf_state);

    products_.publish(parallax::core::make_product(
        parallax::core::ProductId::SpatialEsdf,
        metadata,
        std::move(published)));

    last_map_revision_ = state.revision;

    return parallax::core::SubmitResult::Submitted;
}

}  // namespace parallax::mapping

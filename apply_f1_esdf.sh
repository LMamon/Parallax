#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path

# ------------------------------------------------------------------
# 1. Spatial ESDF state
# ------------------------------------------------------------------

Path("include/parallax/mapping/spatial_esdf_state.hpp").write_text(r'''#pragma once

#include <cstddef>
#include <cstdint>

namespace parallax::mapping {

struct SpatialEsdfState {
    std::uint64_t localization_epoch = 0;
    std::uint64_t map_revision = 0;
    std::uint64_t esdf_revision = 0;
    std::size_t allocated_blocks = 0;
    float voxel_size_m = 0.0F;

    [[nodiscard]] bool valid() const noexcept {
        return localization_epoch > 0 &&
               map_revision > 0 &&
               esdf_revision > 0 &&
               voxel_size_m > 0.0F;
    }
};

}  // namespace parallax::mapping
''')

# ------------------------------------------------------------------
# 2. ESDF producer
# ------------------------------------------------------------------

Path("include/parallax/mapping/esdf_producer.hpp").write_text(r'''#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/mapping/mapping_config.hpp>
#include <parallax/mapping/spatial_map.hpp>
#include <parallax/mapping/spatial_map_state.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace parallax::mapping {

class EsdfProducer final : public parallax::core::Producer {
public:
    EsdfProducer(const MappingConfig& config,
                 SpatialMap& map,
                 parallax::core::ProductStore& products);

    std::string_view name() const noexcept override;
    const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
    const std::vector<parallax::core::ProductId>& outputs() const noexcept override;
    parallax::core::ExecutionPolicy execution_policy() const noexcept override;
    parallax::core::SubmitResult submit(
        parallax::core::ExecutionContext& context) override;

private:
    const MappingConfig& config_;
    SpatialMap& map_;
    parallax::core::ProductStore& products_;

    std::optional<std::uint64_t> last_map_revision_;
    std::uint64_t esdf_revision_ = 0;
    std::uint64_t epoch_ = 0;

    const std::vector<parallax::core::ProductId> inputs_{
        parallax::core::ProductId::SpatialMapState};

    const std::vector<parallax::core::ProductId> outputs_{
        parallax::core::ProductId::SpatialEsdf};
};

}  // namespace parallax::mapping
''')

Path("src/mapping/esdf_producer.cpp").write_text(r'''#include <parallax/mapping/esdf_producer.hpp>

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
''')

# ------------------------------------------------------------------
# 3. ProductId
# ------------------------------------------------------------------

p = Path("include/parallax/core/product_id.hpp")
s = p.read_text()

old = '''        SpatialMapState,
        SpatialTsdf,
        SpatialMesh'''

new = '''        SpatialMapState,
        SpatialTsdf,
        SpatialEsdf,
        SpatialMesh'''

if old not in s:
    raise SystemExit("ProductId insertion point not found")

p.write_text(s.replace(old, new, 1))

# ------------------------------------------------------------------
# 4. Mapping library
# ------------------------------------------------------------------

p = Path("src/mapping/CMakeLists.txt")
s = p.read_text()

old = '''    spatial_tsdf_producer.cpp
    tsdf_snapshot_producer.cpp'''

new = '''    spatial_tsdf_producer.cpp
    tsdf_snapshot_producer.cpp
    esdf_producer.cpp'''

if old not in s:
    raise SystemExit("mapping CMake insertion point not found")

p.write_text(s.replace(old, new, 1))

# ------------------------------------------------------------------
# 5. Runtime ownership
# ------------------------------------------------------------------

p = Path("include/parallax/core/runtime.hpp")
s = p.read_text()

old = '''#include <parallax/mapping/tsdf_snapshot_producer.hpp>
#include <parallax/mapping/spatial_mesh_producer.hpp>'''

new = '''#include <parallax/mapping/tsdf_snapshot_producer.hpp>
#include <parallax/mapping/esdf_producer.hpp>
#include <parallax/mapping/spatial_mesh_producer.hpp>'''

if old not in s:
    raise SystemExit("runtime.hpp include insertion point not found")

s = s.replace(old, new, 1)

old = '''            std::unique_ptr<parallax::mapping::TsdfSnapshotProducer> tsdf_snapshot_producer_;
            std::unique_ptr<parallax::mapping::SpatialMeshProducer> spatial_mesh_producer_;'''

new = '''            std::unique_ptr<parallax::mapping::TsdfSnapshotProducer> tsdf_snapshot_producer_;
            std::unique_ptr<parallax::mapping::EsdfProducer> esdf_producer_;
            std::unique_ptr<parallax::mapping::SpatialMeshProducer> spatial_mesh_producer_;'''

if old not in s:
    raise SystemExit("runtime.hpp producer insertion point not found")

p.write_text(s.replace(old, new, 1))

# ------------------------------------------------------------------
# 6. Runtime construction + graph registration
# ------------------------------------------------------------------

p = Path("src/core/runtime_bootstrap.cpp")
s = p.read_text()

old = '''        tsdf_snapshot_producer_ = std::make_unique<parallax::mapping::TsdfSnapshotProducer>(
            mapping_config_, *spatial_map_, context_.products());

        spatial_mesh_producer_ = std::make_unique<parallax::mapping::SpatialMeshProducer>('''

new = '''        tsdf_snapshot_producer_ = std::make_unique<parallax::mapping::TsdfSnapshotProducer>(
            mapping_config_, *spatial_map_, context_.products());

        esdf_producer_ = std::make_unique<parallax::mapping::EsdfProducer>(
            mapping_config_, *spatial_map_, context_.products());

        spatial_mesh_producer_ = std::make_unique<parallax::mapping::SpatialMeshProducer>('''

if old not in s:
    raise SystemExit("runtime construction insertion point not found")

s = s.replace(old, new, 1)

old = '''        graph_.register_producer(*spatial_tsdf_producer_);
        graph_.register_producer(*tsdf_snapshot_producer_);
        graph_.register_producer(*spatial_mesh_producer_);'''

new = '''        graph_.register_producer(*spatial_tsdf_producer_);
        graph_.register_producer(*tsdf_snapshot_producer_);
        graph_.register_producer(*esdf_producer_);
        graph_.register_producer(*spatial_mesh_producer_);'''

if old not in s:
    raise SystemExit("runtime registration insertion point not found")

p.write_text(s.replace(old, new, 1))

# ------------------------------------------------------------------
# 7. Unit test
# ------------------------------------------------------------------

Path("tests/spatial_esdf_test.cpp").write_text(r'''#include <parallax/mapping/spatial_esdf_state.hpp>

#include <gtest/gtest.h>

TEST(SpatialEsdfStateTest, RequiresVersionedMappedField) {
    parallax::mapping::SpatialEsdfState state;

    EXPECT_FALSE(state.valid());

    state.localization_epoch = 2;
    state.map_revision = 8;
    state.esdf_revision = 5;
    state.allocated_blocks = 12;
    state.voxel_size_m = 0.10F;

    EXPECT_TRUE(state.valid());

    state.voxel_size_m = 0.0F;
    EXPECT_FALSE(state.valid());
}
''')

p = Path("tests/CMakeLists.txt")
s = p.read_text()

needle = '''parallax_add_gtest(
    parallax_mapping_config_test
    mapping_config_test.cpp
    parallax_mapping
)
'''

addition = needle + '''
parallax_add_gtest(
    parallax_spatial_esdf_test
    spatial_esdf_test.cpp
    parallax_mapping
    parallax_core
)
'''

if needle not in s:
    raise SystemExit("tests CMake insertion point not found")

p.write_text(s.replace(needle, addition, 1))

PY

echo
echo "F1 ESDF changes applied."
echo
git diff --stat
git diff --check

#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/mapping/mapping_config.hpp>
#include <parallax/mapping/spatial_map.hpp>
#include <parallax/mapping/spatial_map_state.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace parallax::mapping {

class TsdfSnapshotProducer final : public parallax::core::Producer {
public:
    TsdfSnapshotProducer(const MappingConfig& config,
                         SpatialMap& map,
                         parallax::core::ProductStore& products);

    std::string_view name() const noexcept override;
    const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
    const std::vector<parallax::core::ProductId>& outputs() const noexcept override;
    parallax::core::ExecutionPolicy execution_policy() const noexcept override;
    parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;

private:
    const MappingConfig& config_;
    SpatialMap& map_;
    parallax::core::ProductStore& products_;
    std::optional<std::uint64_t> last_revision_;

    const std::vector<parallax::core::ProductId> inputs_{
        parallax::core::ProductId::SpatialMapState};
    const std::vector<parallax::core::ProductId> outputs_{
        parallax::core::ProductId::SpatialTsdf};
};

}  // namespace parallax::mapping

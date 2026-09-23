#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/mapping/mapping_config.hpp>
#include <parallax/mapping/spatial_map.hpp>
#include <parallax/mapping/spatial_map_state.hpp>
#include <parallax/stereo/calibration.hpp>

#include <nvblox/sensors/camera.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace parallax::mapping {

class SpatialMeshProducer final : public parallax::core::Producer {
public:
    SpatialMeshProducer(const MappingConfig& config,
                        const parallax::stereo::StereoCalibration& calibration,
                        SpatialMap& map,
                        parallax::core::ProductStore& products);

    std::string_view name() const noexcept override;
    const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
    const std::vector<parallax::core::ProductId>& outputs() const noexcept override;
    const std::vector<parallax::core::CompatibleInputRequirement>& compatible_inputs() const noexcept override;
    parallax::core::ExecutionPolicy execution_policy() const noexcept override;
    parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;

private:
    static constexpr std::size_t RectifiedRgbHistoryCapacity = 4;

    const MappingConfig& config_;
    SpatialMap& map_;
    parallax::core::ProductStore& products_;
    nvblox::Camera camera_;
    std::optional<std::uint64_t> last_revision_;
    std::uint64_t mesh_revision_ = 0;

    const std::vector<parallax::core::ProductId> inputs_{
        parallax::core::ProductId::SpatialMapState,
        parallax::core::ProductId::RectifiedRgb};
    const std::vector<parallax::core::ProductId> outputs_{
        parallax::core::ProductId::SpatialMesh};
    const std::vector<parallax::core::CompatibleInputRequirement> compatible_inputs_{
        {parallax::core::ProductId::RectifiedRgb, RectifiedRgbHistoryCapacity}};
};

}  // namespace parallax::mapping

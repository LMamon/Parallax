#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/mapping/mapping_config.hpp>
#include <parallax/mapping/spatial_map.hpp>
#include <parallax/mapping/spatial_map_state.hpp>
#include <parallax/core/sensor_extrinsics.hpp>
#include <parallax/localization/localization.hpp>
#include <parallax/stereo/calibration.hpp>

#include <nvblox/sensors/camera.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace parallax::mapping {

// Persistent-map integrator. Despite the historical class name, this producer
// no longer materializes a TSDF visualization product. It only advances the
// GPU-resident SpatialMap and publishes a lightweight map revision.
class SpatialTsdfProducer final : public parallax::core::Producer {
public:
    SpatialTsdfProducer(const parallax::stereo::StereoCalibration& calibration,
                        const parallax::core::SensorExtrinsics& extrinsics,
                        const MappingConfig& config,
                        SpatialMap& map,
                        parallax::core::ProductStore& products);

    std::string_view name() const noexcept override;
    const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
    const std::vector<parallax::core::ProductId>& outputs() const noexcept override;
    const std::vector<parallax::core::CompatibleInputRequirement>& compatible_inputs() const noexcept override;
    parallax::core::ExecutionPolicy execution_policy() const noexcept override;
    parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;

private:
    static constexpr std::size_t DepthHistoryCapacity = 4;

    void resetForEpoch(std::uint64_t epoch);
    nvblox::Transform worldFromRectifiedCamera(
        const parallax::localization::LocalizationPose& pose) const;

    const parallax::stereo::StereoCalibration& calibration_;
    const MappingConfig& config_;
    SpatialMap& map_;
    parallax::core::ProductStore& products_;
    nvblox::Camera camera_;

    std::array<double, 9> body_from_rectified_rotation_{};
    std::array<double, 3> body_from_rectified_translation_{};

    std::optional<std::uint64_t> epoch_;
    std::optional<parallax::core::SourceObservation> last_integrated_;
    std::uint64_t integrated_frames_ = 0;
    std::uint64_t epoch_resets_ = 0;
    std::uint64_t map_revision_ = 0;

    const std::vector<parallax::core::ProductId> inputs_{
        parallax::core::ProductId::LocalizationPose,
        parallax::core::ProductId::Depth};
    const std::vector<parallax::core::ProductId> outputs_{
        parallax::core::ProductId::SpatialMapState};
    const std::vector<parallax::core::CompatibleInputRequirement> compatible_inputs_{
        {parallax::core::ProductId::Depth, DepthHistoryCapacity}};
};

}  // namespace parallax::mapping

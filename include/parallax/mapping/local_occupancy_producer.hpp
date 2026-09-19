#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/core/dependency_resolver.hpp>
#include <parallax/mapping/local_occupancy_state.hpp>
#include <parallax/core/sensor_extrinsics.hpp>
#include <parallax/localization/localization.hpp>
#include <parallax/stereo/calibration.hpp>

#include <nvblox/core/cuda_stream.h>
#include <nvblox/mapper/mapper.h>
#include <nvblox/sensors/camera.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace parallax::mapping {

class LocalOccupancyProducer final : public parallax::core::Producer {
public:
    LocalOccupancyProducer(const parallax::stereo::StereoCalibration& calibration,
                           const parallax::core::SensorExtrinsics& extrinsics,
                           parallax::core::ProductStore& products,
                           const parallax::core::DependencyResolver& resolver,
                           cudaStream_t cuda_stream);

    std::string_view name() const noexcept override;
    const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
    const std::vector<parallax::core::ProductId>& outputs() const noexcept override;
    const std::vector<parallax::core::CompatibleInputRequirement>& compatible_inputs() const noexcept override;
    parallax::core::ExecutionPolicy execution_policy() const noexcept override;
    parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;

private:
    static constexpr float VoxelSizeM = 0.15F;
    static constexpr float KeepRadiusM = 10.0F;
    static constexpr std::size_t DepthHistoryCapacity = 4;
    // 8.1 m x 8.1 m x 4.05 m bounded visualization snapshot.
    static constexpr std::uint32_t SnapshotColumns = 54;
    static constexpr std::uint32_t SnapshotRows = 54;
    static constexpr std::uint32_t SnapshotSlices = 27;

    void resetForEpoch(std::uint64_t epoch);
    bool buildSnapshot(const nvblox::Vector3f& center, LocalOccupancyState* state);
    nvblox::Transform worldFromRectifiedCamera(
        const parallax::localization::LocalizationPose& pose) const;

    const parallax::stereo::StereoCalibration& calibration_;
    parallax::core::ProductStore& products_;
    const parallax::core::DependencyResolver& resolver_;
    cudaStream_t cuda_stream_ = nullptr;
    std::shared_ptr<nvblox::CudaStreamNonOwning> nvblox_stream_;
    nvblox::Camera camera_;
    std::unique_ptr<nvblox::Mapper> mapper_;

    std::array<double, 9> body_from_rectified_rotation_{};
    std::array<double, 3> body_from_rectified_translation_{};

    std::optional<std::uint64_t> epoch_;
    std::optional<parallax::core::SourceObservation> last_integrated_;
    std::uint64_t integrated_frames_ = 0;
    std::uint64_t epoch_resets_ = 0;

    const std::vector<parallax::core::ProductId> inputs_{
        parallax::core::ProductId::LocalizationPose,
        parallax::core::ProductId::Depth
    };
    const std::vector<parallax::core::ProductId> outputs_{
        parallax::core::ProductId::LocalOccupancy
    };
    const std::vector<parallax::core::CompatibleInputRequirement> compatible_inputs_{
        {parallax::core::ProductId::Depth, DepthHistoryCapacity}
    };
};

}  // namespace parallax::mapping

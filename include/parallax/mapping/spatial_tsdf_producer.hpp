#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/core/dependency_resolver.hpp>
#include <parallax/mapping/spatial_tsdf_state.hpp>
#include <parallax/mapping/spatial_mesh_state.hpp>
#include <parallax/mapping/mapping_config.hpp>
#include <parallax/core/sensor_extrinsics.hpp>
#include <parallax/localization/localization.hpp>
#include <parallax/stereo/calibration.hpp>

#include <nvblox/core/cuda_stream.h>
#include <nvblox/mapper/mapper.h>
#include <nvblox/sensors/camera.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace parallax::mapping {

    class SpatialTsdfProducer final : public parallax::core::Producer {
        public:
            SpatialTsdfProducer(const parallax::stereo::StereoCalibration& calibration,
                                   const parallax::core::SensorExtrinsics& extrinsics,
                                   const MappingConfig& config,
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
            static constexpr std::size_t DepthHistoryCapacity = 4;
            static constexpr std::size_t RectifiedRgbHistoryCapacity = 4;

            void resetForEpoch(std::uint64_t epoch);

            nvblox::Transform worldFromRectifiedCamera(const parallax::localization::LocalizationPose& pose) const;

            const parallax::stereo::StereoCalibration& calibration_;
            const MappingConfig& config_;
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
            std::optional<parallax::core::SourceObservation> last_color_integrated_;
            std::chrono::steady_clock::time_point last_color_integration_{};
            std::chrono::steady_clock::time_point last_mesh_update_{};
            
            std::uint64_t integrated_frames_ = 0;
            std::uint64_t epoch_resets_ = 0;
            std::uint64_t mesh_revision_ = 0;

            const std::vector<parallax::core::ProductId> inputs_{
                parallax::core::ProductId::LocalizationPose, parallax::core::ProductId::Depth,
                parallax::core::ProductId::RectifiedRgb
            };
            const std::vector<parallax::core::ProductId> outputs_{
                parallax::core::ProductId::SpatialTsdf, parallax::core::ProductId::SpatialMesh
            };
            const std::vector<parallax::core::CompatibleInputRequirement> compatible_inputs_{
                {parallax::core::ProductId::Depth, DepthHistoryCapacity},
                {parallax::core::ProductId::RectifiedRgb, RectifiedRgbHistoryCapacity}
            };
    };
}

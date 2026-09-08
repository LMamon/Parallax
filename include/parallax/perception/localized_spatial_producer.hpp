#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/core/sensor_extrinsics.hpp>
#include <parallax/stereo/calibration.hpp>

#include <array>
#include <chrono>
#include <vector>

namespace parallax::perception {

    class LocalizedSpatialProducer final : public core::Producer {
        public:
            LocalizedSpatialProducer(const stereo::StereoCalibration& calibration,
                                     const core::SensorExtrinsics& extrinsics,
                                     core::ProductStore& products);

            [[nodiscard]] std::string_view name() const noexcept override;
            [[nodiscard]] const std::vector<core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<core::ProductId>& outputs() const noexcept override;

            [[nodiscard]] const std::vector<core::CompatibleInputRequirement>&
            compatible_inputs() const noexcept override;

            [[nodiscard]] core::ExecutionPolicy execution_policy() const noexcept override;

            core::SubmitResult submit(core::ExecutionContext& context) override;

        private:
            static constexpr std::size_t LocalizationHistoryCapacity = 16;
            static constexpr auto MaxPoseDelta = std::chrono::milliseconds{50};

            std::array<double, 9> body_from_rectified_rotation_{};
            std::array<double, 3> body_from_rectified_translation_{};

            core::ProductStore& products_;

            const std::vector<core::ProductId> inputs_{core::ProductId::Object3D, core::ProductId::LocalizationPose};
            const std::vector<core::ProductId> outputs_{core::ProductId::LocalizedSpatialObservation};

            // causes the existing history configuration to provision bounded LocalizationPose 
            // history instead of making this producer manage its own pose cache.
            const std::vector<core::CompatibleInputRequirement> compatible_inputs_{{core::ProductId::LocalizationPose,
                                                                                    LocalizationHistoryCapacity}};
    };

}
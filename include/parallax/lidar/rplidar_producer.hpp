#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/lidar/rplidar.hpp>

#include <cstdint>
#include <vector>

namespace parallax::lidar {
    /**
     * LiDAR has no camera dependency. Consumers that later need both sensor
     * families must perform compatibility/freshness selection explicitly
     * rather than forcing acquisition lockstep here.
     */
    class RplidarSourceProducer final : public parallax::core::Producer {
        public:
            RplidarSourceProducer(Rplidar& lidar, parallax::core::ProductStore& store);
            [[nodiscard]] std::string_view name() const noexcept override;
            
            [[nodiscard]] const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::ProductId>& outputs() const noexcept override;

            [[nodiscard]] parallax::core::ExecutionPolicy execution_policy() const noexcept override;
        
            parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;

        private:
            Rplidar& lidar_;
            parallax::core::ProductStore& store_;

            std::uint64_t sequence_{0};

            const std::vector<parallax::core::ProductId> inputs_{};
            const std::vector<parallax::core::ProductId> outputs_{
                parallax::core::ProductId::LidarScan
            };
    };
}
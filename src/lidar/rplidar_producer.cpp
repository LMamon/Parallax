#include <parallax/lidar/rplidar_producer.hpp>
#include <parallax/core/product.hpp>

#include <iostream>
#include <chrono>
#include <memory>
#include <utility>

namespace parallax::lidar {

    RplidarSourceProducer::RplidarSourceProducer(Rplidar& lidar, parallax::core::ProductStore& store) :
                                                 lidar_(lidar),
                                                 store_(store) {}


    std::string_view RplidarSourceProducer::name() const noexcept {
        return "rplidar";
    }

    const std::vector<parallax::core::ProductId>& RplidarSourceProducer::inputs() const noexcept {
        return inputs_;
    }


    const std::vector<parallax::core::ProductId>& RplidarSourceProducer::outputs() const noexcept {
        return outputs_;
    }

    parallax::core::ExecutionPolicy RplidarSourceProducer::execution_policy() const noexcept {
        /**
         * Scan acquisition and SDK interaction are CPU-side. Detailed target
         * rates and scheduling policy land later with the execution-policy
         * update this declaration only records the resource affinity.
         */
        parallax::core::ExecutionPolicy policy{};
        policy.affinity = parallax::core::ResourceAffinity::Cpu;
        policy.stateful = false;
        return policy;
    }


    parallax::core::SubmitResult RplidarSourceProducer::submit(parallax::core::ExecutionContext& context) {
        (void)context;
        auto scan = std::make_shared<LidarScan>();

        if (!lidar_.capture(*scan)) {
            return parallax::core::SubmitResult::Failed;
        }
        
        /**
         * This timestamp describes the LiDAR observation boundary itself.
         * It is intentionally not copied from the latest camera frame:
         * independent sensor clocks/cadences remain independent until a
         * consumer explicitly requests cross-sensor compatibility.
         */
        const auto now = std::chrono::steady_clock::now();
        parallax::core::ProductMetadata metadata{};
        metadata.observation.source = parallax::core::SourceId::Rplidar;
        metadata.observation.sequence = ++sequence_;
        metadata.timestamp = now;
        metadata.production_timestamp = now;

        metadata.valid = scan->valid();

        if (!metadata.valid) {
            return parallax::core::SubmitResult::NoWork;
        }
    
        // Product payloads become immutable once published. Finish building the
        // scan first, then explicitly convert to the const shared-pointer type
        // expected by make_product().
        std::shared_ptr<const LidarScan> published_scan = std::move(scan);

       store_.publish(parallax::core::make_product(parallax::core::ProductId::LidarScan, 
                                                   metadata, 
                                                   std::move(published_scan)));

        return parallax::core::SubmitResult::Submitted;
    }
}
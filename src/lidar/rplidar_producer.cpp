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
        parallax::core::ExecutionPolicy policy;

        /**
         * Scan acquisition and SDK interaction are CPU-side. Detailed target
         * rates and scheduling policy land later with the execution-policy
         * update this declaration only records the resource affinity.
         */
        policy.affinity = parallax::core::ResourceAffinity::Cpu;

        return policy;
    }


    parallax::core::SubmitResult RplidarSourceProducer::submit() {
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
        parallax::core::ProductMetadata metadata;
        metadata.sequence = ++sequence_;
        metadata.timestamp = std::chrono::steady_clock::now();
        metadata.valid = scan->valid();

        if (!metadata.valid) {
            return parallax::core::SubmitResult::NoWork;
        }
        
        const std::size_t point_count = scan->points.size();
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
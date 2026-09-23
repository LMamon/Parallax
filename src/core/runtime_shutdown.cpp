#include <parallax/core/runtime.hpp>
#include <parallax/core/pipeline.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/pose/charuco_pose.hpp>
#include <parallax/core/history_configuration.hpp>
#include <parallax/core/runtime_metrics.hpp>
#include <parallax/mapping/mapping_metrics.hpp>
#include <parallax/application/foxglove_command.hpp>
#include <parallax/camera/arducam_controls.hpp>

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string_view>
#include <limits>
#include <vector>
#include <iostream>

namespace parallax::core {
void Runtime::stop() noexcept { running_.store(false); }

void Runtime::shutdown() {
        stop();

        if (auto_control_thread_.joinable()) auto_control_thread_.join();
        if (lidar_thread_.joinable()) lidar_thread_.join();
        if (visualization_thread_.joinable()) visualization_thread_.join();

        // Stop physical hardware as soon as its worker can no longer access it.
        if (lidar_) lidar_->shutdown();

        if (!context_.drain()) std::cerr << "Runtime: failed to drain execution context during shutdown\n";

        publisher_.shutdown();
        foxglove_.shutdown();

        if (single_target_producer_) single_target_producer_->reset();

        request_controller_.reset();

        context_.products().clear();
        producer_execution_state_.clear();

        marker_depth_producer_.reset();
        charuco_pose_producer_.reset();
        depth_producer_.reset();
        stereo_producer_.reset();
        rectification_producer_.reset();
        isp_producer_.reset();
        auto_controller_.reset();
        lidar_producer_.reset();
        camera_producer_.reset();
        segmentation_producer_.reset();
        single_target_producer_.reset();

        if (efficientvit_sam_) efficientvit_sam_->shutdown();
        efficientvit_sam_.reset();
        detection_producer_.reset();
        cuvslam_producer_.reset();

        if (cuvslam_localizer_) {
            cuvslam_localizer_->shutdown();
            cuvslam_localizer_.reset();
        }

        // Do NOT explicitly shutdown/reset nanoowl_ here.
        // Its lifetime remains owned by Runtime.
        pipeline_.shutdown();
        lidar_.reset();

        if (camera_) {
            camera_->shutdown();
            camera_.reset();
        }

        context_.shutdown();
        initialized_ = false;
    }
}

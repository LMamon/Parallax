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
void Runtime::runVisualization() {
        using namespace std::chrono_literals;

        while (running_.load()) {
            if (foxglove_.takeCalibrationRequest() &&
                (!publisher_.publishLeftCalibration(pipeline_.calibration()) ||
                 !publisher_.publishDepthCalibration(pipeline_.calibration()))) {
                visualization_failed_.store(true);
                running_.store(false);
                return;
            }

            if (foxglove_.takeTransformRequest() &&
                !publisher_.publishStaticTransforms(sensor_extrinsics_)) {
                visualization_failed_.store(true);
                running_.store(false);
                return;
            }

            if (!publisher_.publishAvailable(
                    context_.products(),
                    [this](const CompletionHandle& completion) {
                        return context_.waitForHost(completion);
                    })) {
                visualization_failed_.store(true);
                running_.store(false);
                return;
            }

            std::this_thread::sleep_for(5ms);
        }
    }

void Runtime::runAutoControl() {
        using namespace std::chrono_literals;
        
        while (running_.load() && auto_controller_) {
            parallax::isp::IspStatistics statistics{};
        
            if (pipeline_.isp().tryGetStatistics(statistics)) {
                const auto update = auto_controller_->update(statistics);
        
                if (update.gain_changed && !camera_->setControl(parallax::camera::controls::AnalogGain, update.analogue_gain)) {
                    std::cerr << "Runtime: automatic gain update failed; disabling auto control\n"; return;
                }
        
                if (update.exposure_changed && !camera_->setControl(parallax::camera::controls::Exposure, update.exposure)) {
                    std::cerr << "Runtime: automatic exposure update failed; disabling auto control\n"; return;
                }
        
                if (update.white_balance_changed) pipeline_.isp().setWhiteBalance(update.white_balance);
            }
            std::this_thread::sleep_for(5ms);
        }
    }

void Runtime::runLidarSource() {
        /**
        * Resolve the independently clocked LiDAR branch once. LidarScan is a
        * source product, so the expected plan contains only the RPLIDAR producer.
        *
        * Keeping this plan on its own worker prevents blocking SLAMTEC SDK reads
        * from setting the cadence of the camera/stereo branch.
        */
        const auto execution_plan = resolver_.resolve(ProductId::LidarScan);
        if (execution_plan.empty()) {
            std::cerr << "Runtime: no producer available for lidarscan\n";
            return;
        }

        while (running_.load()) {
            for (Producer* producer : execution_plan) {
                if (producer == nullptr) {
                    std::cerr << "Runtime: null producer in lidar execution plan\n";
                    return;
                }

                auto& stats = producer_execution_stats_.at(producer);
                ++stats.considered;

                const SubmitResult result = producer->submit(context_);
                if (result == SubmitResult::Failed) {
                    ++stats.failed;

                    std::cerr << "Runtime: lidar producer failed: " << producer->name() << '\n';
                    return;
                }

                if (result == SubmitResult::NoWork) {
                    ++stats.no_work;
                    continue;
                }

                ++stats.submitted;
            }
        }
    }
}

#include <parallax/camera/stereo_node.hpp>

namespace parallax::ros {

	void StereoNode::initializeAutoControl() {
		if (!isp_config_.auto_exposure.enable && !isp_config_.auto_white_balance.enable) {
			return;
		}

		parallax::camera::ControlRange exposure_range{};
		parallax::camera::ControlRange gain_range{};

		if (isp_config_.auto_exposure.enable) {

			if (!camera_->getControlRange(parallax::camera::controls::Exposure, exposure_range) ||
				!camera_->getControlRange(parallax::camera::controls::AnalogGain, gain_range)) {
			    
                throw std::runtime_error("failed to query exposure/gain control ranges");
			}
		} else {
			exposure_range = {static_cast<std::int32_t>(camera_config_.exposure),
                             static_cast<std::int32_t>(camera_config_.exposure),
                             1,
                             static_cast<std::int32_t>(camera_config_.exposure),
                             true};

			gain_range = {static_cast<std::int32_t>(camera_config_.analogue_gain),
                          static_cast<std::int32_t>(camera_config_.analogue_gain),
                          1,
                          static_cast<std::int32_t>(camera_config_.analogue_gain),
                          true};
		}

		auto_controller_ = std::make_unique<parallax::isp::AutoController>(
                                            isp_config_,
                                            exposure_range,
                                            gain_range,
                                            static_cast<std::int32_t>(camera_config_.exposure),
                                            static_cast<std::int32_t>(camera_config_.analogue_gain));
    }

    void StereoNode::autoControlLoop() {
        using namespace std::chrono_literals;

        while (running_.load() && rclcpp::ok() && auto_controller_) {
            parallax::isp::IspStatistics statistics{};
            if (isp_.tryGetStatistics(statistics)) {
                const auto update = auto_controller_->update(statistics);
                if (update.gain_changed &&
                    !camera_->setControl(parallax::camera::controls::AnalogGain, update.analogue_gain)) {
                    RCLCPP_ERROR(get_logger(), "automatic gain update failed; disabling auto control");
                    return;
                }
                if (update.exposure_changed &&
                    !camera_->setControl(parallax::camera::controls::Exposure, update.exposure)) {
                    RCLCPP_ERROR(get_logger(), "automatic exposure update failed; disabling auto control");
                    return;
                }
                if (update.white_balance_changed) isp_.setWhiteBalance(update.white_balance);
            }
            std::this_thread::sleep_for(5ms);
        }
    }

}

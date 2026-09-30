#include <parallax/camera/stereo_node.hpp>

namespace parallax::ros {

void StereoNode::initializeAutoControl() {
  if (!isp_config_.auto_exposure.enable &&
      !isp_config_.auto_white_balance.enable) {
    return;
  }

  parallax::camera::ControlRange exposure_range{};
  parallax::camera::ControlRange gain_range{};

  if (isp_config_.auto_exposure.enable) {
    if (!camera_->getControlRange(
            parallax::camera::controls::Exposure, exposure_range) ||
        !camera_->getControlRange(
            parallax::camera::controls::AnalogGain, gain_range)) {
      throw std::runtime_error(
          "failed to query exposure/gain control ranges");
    }
  } else {
    exposure_range = {
        static_cast<std::int32_t>(camera_config_.exposure),
        static_cast<std::int32_t>(camera_config_.exposure),
        1,
        static_cast<std::int32_t>(camera_config_.exposure),
        true,
    };
    gain_range = {
        static_cast<std::int32_t>(camera_config_.analogue_gain),
        static_cast<std::int32_t>(camera_config_.analogue_gain),
        1,
        static_cast<std::int32_t>(camera_config_.analogue_gain),
        true,
    };
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
  auto diag_start = std::chrono::steady_clock::now();
  std::uint64_t diag_stats = 0;
  std::uint64_t diag_exposure = 0;
  std::uint64_t diag_gain = 0;
  std::uint64_t diag_wb = 0;

  while (running_.load() && rclcpp::ok() && auto_controller_) {
    parallax::isp::IspStatistics statistics{};

    if (isp_.tryGetStatistics(statistics)) {
      ++diag_stats;
      const auto update = auto_controller_->update(statistics);
      if (update.exposure_changed) ++diag_exposure;
      if (update.gain_changed) ++diag_gain;
      if (update.white_balance_changed) ++diag_wb;

      if (update.gain_changed &&
          !camera_->setControl(
              parallax::camera::controls::AnalogGain,
              update.analogue_gain)) {
        RCLCPP_ERROR(
            get_logger(),
            "automatic gain update failed; disabling auto control");
        return;
      }

      if (update.exposure_changed &&
          !camera_->setControl(
              parallax::camera::controls::Exposure,
              update.exposure)) {
        RCLCPP_ERROR(
            get_logger(),
            "automatic exposure update failed; disabling auto control");
        return;
      }

      if (update.white_balance_changed) {
        isp_.setWhiteBalance(update.white_balance);
      }
    }

    const auto diag_now = std::chrono::steady_clock::now();
    if (diagnostics_ &&
        diag_now - diag_start >= std::chrono::seconds(5)) {
      const double seconds =
          std::chrono::duration<double>(diag_now - diag_start).count();
      RCLCPP_INFO(
          get_logger(),
          "camera_diag auto_stats=%.2fHz exposure=%llu gain=%llu wb=%llu",
          static_cast<double>(diag_stats) / seconds,
          static_cast<unsigned long long>(diag_exposure),
          static_cast<unsigned long long>(diag_gain),
          static_cast<unsigned long long>(diag_wb));
      diag_stats = diag_exposure = diag_gain = diag_wb = 0;
      diag_start = diag_now;
    }

    std::this_thread::sleep_for(5ms);
  }
}

}  // namespace parallax::ros

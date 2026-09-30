#include <parallax/camera/stereo_node.hpp>

namespace parallax::ros {

void StereoNode::acquisitionLoop() {
  unsigned failures = 0;
  std::uint64_t sequence = 0;

  auto boundary_window_start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::duration boundary_capture_time{};
  std::chrono::steady_clock::duration boundary_post_capture_time{};
  std::uint64_t boundary_frames = 0;
  auto diag_start = std::chrono::steady_clock::now();
  std::uint64_t diag_frames = 0;
  std::uint64_t diag_superseded = 0;

  while (running_.load() && rclcpp::ok()) {
    parallax::camera::RawFrame frame{};
    const auto boundary_capture_start = std::chrono::steady_clock::now();
    if (!camera_->capture(frame)) {
      if (++failures >= 10) {
        RCLCPP_ERROR(get_logger(), "camera repeatedly failed to capture");
        running_.store(false);
        raw_cv_.notify_all();
        return;
      }
      continue;
    }

    const auto boundary_capture_end = std::chrono::steady_clock::now();
    failures = 0;

    const std::size_t expected_bytes =
        static_cast<std::size_t>(camera_config_.width) *
        static_cast<std::size_t>(camera_config_.height) *
        sizeof(std::uint16_t);
    if (frame.data == nullptr || frame.device_data == nullptr ||
        frame.bytes < expected_bytes) {
      (void)camera_->release(frame);
      RCLCPP_ERROR(get_logger(), "camera returned an invalid mapped BA10 frame");
      continue;
    }

    const auto wall_timestamp = monotonicToSystemTime(frame.timestamp);

    RawLease superseded{};
    {
      std::lock_guard<std::mutex> lock(raw_mutex_);
      if (raw_pending_.valid) {
        superseded = raw_pending_;
        ++diag_superseded;
      }
      raw_pending_.frame = frame;
      raw_pending_.sequence = ++sequence;
      raw_pending_.wall_timestamp = wall_timestamp;
      raw_pending_.valid = true;
    }

    // A pending frame that compute never claimed has no GPU readers and can
    // return to VI immediately. The newest frame remains the mailbox value.
    if (superseded.valid && !camera_->release(superseded.frame)) {
      RCLCPP_ERROR(get_logger(), "failed to requeue superseded camera buffer");
      running_.store(false);
      raw_cv_.notify_all();
      return;
    }
    raw_cv_.notify_one();

    const auto boundary_iteration_end = std::chrono::steady_clock::now();
    boundary_capture_time += boundary_capture_end - boundary_capture_start;
    boundary_post_capture_time += boundary_iteration_end - boundary_capture_end;
    ++boundary_frames;
    ++diag_frames;

    const auto boundary_window = boundary_iteration_end - boundary_window_start;
    if (diagnostics_ && boundary_window >= std::chrono::seconds(5) &&
        boundary_frames != 0) {
      const double window_s =
          std::chrono::duration<double>(boundary_window).count();
      const double capture_ms =
          std::chrono::duration<double, std::milli>(boundary_capture_time).count() /
          static_cast<double>(boundary_frames);
      const double post_ms =
          std::chrono::duration<double, std::milli>(boundary_post_capture_time).count() /
          static_cast<double>(boundary_frames);
      RCLCPP_INFO(
          get_logger(),
          "camera_boundary dqbuf=%.3fms/frame post_capture=%.3fms/frame "
          "completed=%.2fHz frames=%llu",
          capture_ms, post_ms,
          static_cast<double>(boundary_frames) / window_s,
          static_cast<unsigned long long>(boundary_frames));
      boundary_window_start = boundary_iteration_end;
      boundary_capture_time = {};
      boundary_post_capture_time = {};
      boundary_frames = 0;
    }

    const auto diag_now = std::chrono::steady_clock::now();
    if (diagnostics_ && diag_now - diag_start >= std::chrono::seconds(5)) {
      const double seconds =
          std::chrono::duration<double>(diag_now - diag_start).count();
      RCLCPP_INFO(
          get_logger(),
          "camera_diag capture=%.2fHz raw_copy=0.000ms/frame superseded=%llu",
          static_cast<double>(diag_frames) / seconds,
          static_cast<unsigned long long>(diag_superseded));
      diag_frames = 0;
      diag_superseded = 0;
      diag_start = diag_now;
    }
  }
}

}  // namespace parallax::ros

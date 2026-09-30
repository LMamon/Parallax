#include <parallax/camera/stereo_node.hpp>
#include <parallax/cuda/depth.cuh>
#include <parallax/cuda/gray_resize.cuh>
#include <parallax/stereo/depth_policy.hpp>

namespace parallax::ros {

void StereoNode::stereoLoop() {
  std::uint64_t completed = 0;
  std::uint64_t superseded = 0;
  auto diag_start = std::chrono::steady_clock::now();

  while (running_.load() && rclcpp::ok()) {
    std::shared_ptr<
        parallax::stereo::StereoRectifier::OutputSlot> rectified;

    parallax::core::CompletionHandle rect_ready{};

    rclcpp::Time stamp(0, 0, RCL_SYSTEM_TIME);

    {
      std::unique_lock<std::mutex> lock(stereo_mutex_);

      stereo_cv_.wait(lock, [&] {
        return !running_.load() ||
               stereo_pending_ != nullptr;
      });

      if (!running_.load()) {
        return;
      }

      rectified = std::move(stereo_pending_);
      stereo_pending_.reset();

      rect_ready = std::move(stereo_pending_ready_);
      stereo_pending_ready_ = {};

      stamp = stereo_pending_stamp_;
    }

    auto& lane = context_.stereoLane();

    if (!context_.waitFor(rect_ready, lane)) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "failed to establish rectifier-to-stereo dependency");

      continue;
    }

    if (!parallax::cuda::downsampleGray2x(
            rectified->gray.left,
            spatial_gray_.left,
            lane.cudaHandle()) ||
        !parallax::cuda::downsampleGray2x(
            rectified->gray.right,
            spatial_gray_.right,
            lane.cudaHandle())) {

      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "gray spatial downsample failed");

      continue;
    }

    auto match =
        matcher_.acquireOutput(context_);

    auto depth =
        depth_pool_.acquire();

    if (!match || !depth) {
      ++superseded;
      continue;
    }

    if (!matcher_.process(
            spatial_gray_,
            *match,
            lane.handle())) {

      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "CUDA stereo matching failed");

      continue;
    }

    const float fx =
        static_cast<float>(
            calibration_.P1()[0]) *
        0.5F;

    const float baseline =
        static_cast<float>(
            calibration_.metadata().baseline_mm /
            1000.0);

    if (!parallax::cuda::disparityToDepth(
            match->output.disparity,
            depth->depth,
            fx,
            baseline,
            parallax::isp::StereoMatchFrame::DisparityScale,
            parallax::stereo::MinUsefulDepthM,
            parallax::stereo::MaxUsefulDepthM,
            lane.cudaHandle())) {

      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "disparity-to-depth failed");

      continue;
    }

    // Publication is still a host observation boundary: the NITROS wrapper
    // carries allocation ownership but not this lane's completion primitive.
    if (!lane.synchronize()) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "stereo lane synchronization failed");

      continue;
    }

    std_msgs::msg::Header header;
    header.stamp = stamp;
    header.frame_id = kLeftFrame;

    try {
      auto depth_image =
          makePooledNitrosImage(
              depth,
              depth->depth,
              header,
              nvidia::gxf::VideoFormat::
                  GXF_VIDEO_FORMAT_D32F,
              "D",
              4U);

      depth_nitros_pub_->publish(depth_image);

    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "NITROS depth publication failed: %s",
          e.what());

      continue;
    }

    ++completed;

    const auto now =
        std::chrono::steady_clock::now();

    if (diagnostics_ &&
        now - diag_start >= std::chrono::seconds(5)) {

      const double seconds =
          std::chrono::duration<double>(
              now - diag_start).count();

      RCLCPP_INFO(
          get_logger(),
          "stereo_diag depth=%.2fHz superseded=%llu "
          "path=Y8->CUDA_SGM->depth",
          static_cast<double>(completed) / seconds,
          static_cast<unsigned long long>(
              superseded));

      completed = 0;
      superseded = 0;
      diag_start = now;
    }
  }
}

}  // namespace parallax::ros

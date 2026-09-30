#include <parallax/camera/stereo_node.hpp>

namespace parallax::ros {

void StereoNode::depthPublishLoop() {
  using Clock = std::chrono::steady_clock;

  auto diag_start = Clock::now();
  std::uint64_t diag_frames = 0;

  Clock::duration diag_ready_wait{};
  Clock::duration diag_nitros_wrap{};
  Clock::duration diag_nitros_publish{};

  while (rclcpp::ok()) {
    DepthPublication publication;

    {
      std::unique_lock<std::mutex> lock(
          depth_mutex_);

      depth_cv_.wait(lock, [&] {
        return !running_.load() ||
               depth_size_ != 0;
      });

      //
      // Drain already-submitted generations during shutdown so their
      // pooled GPU resources are not released while work is in flight.
      //
      if (depth_size_ == 0) {
        if (!running_.load()) {
          return;
        }

        continue;
      }

      publication =
          std::move(
              depth_queue_[depth_head_]);

      depth_queue_[depth_head_] =
          DepthPublication{};

      depth_head_ =
          (depth_head_ + 1) %
          DepthSlotCount;

      --depth_size_;
    }

    const auto ready_start = Clock::now();

    const bool depth_ready =
        context_.waitForHost(
            publication.ready);

    diag_ready_wait +=
        Clock::now() - ready_start;

    if (!depth_ready) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "failed waiting for depth readiness");

      publication.spatial.reset();
      publication.match.reset();
      publication.depth.reset();
      continue;
    }

    //
    // The complete stereo generation has retired. Neither SGM nor the
    // depth conversion can reference these inputs anymore.
    //
    publication.spatial.reset();
    publication.match.reset();

    //
    // As with RGB, retire safely during shutdown but do not create new
    // ROS publications.
    //
    if (!running_.load()) {
      publication.depth.reset();
      continue;
    }

    const auto wrap_start = Clock::now();

    try {
      auto depth_image =
          makePooledNitrosImage(
              publication.depth,
              publication.depth->depth,
              publication.header,
              nvidia::gxf::VideoFormat::
                  GXF_VIDEO_FORMAT_D32F,
              "D",
              4U);

      const auto wrap_end = Clock::now();

      depth_nitros_pub_->publish(
          depth_image);

      const auto publish_end = Clock::now();

      diag_nitros_wrap +=
          wrap_end - wrap_start;

      diag_nitros_publish +=
          publish_end - wrap_end;

    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "NITROS depth publication failed: %s",
          e.what());

      continue;
    }

    ++diag_frames;

    const auto now = Clock::now();

    if (diagnostics_ &&
        now - diag_start >=
            std::chrono::seconds(5)) {

      const double seconds =
          std::chrono::duration<double>(
              now - diag_start).count();

      const auto ms_per_frame =
          [diag_frames](
              Clock::duration duration) -> double {

        if (diag_frames == 0) {
          return 0.0;
        }

        return
            std::chrono::duration<
                double,
                std::milli>(duration).count() /
            static_cast<double>(diag_frames);
      };

      std::size_t queue_depth = 0;

      {
        std::lock_guard<std::mutex> lock(
            depth_mutex_);

        queue_depth = depth_size_;
      }

      RCLCPP_INFO(
          get_logger(),
          "depth_profile rate=%.2fHz "
          "queue=%zu/%zu "
          "ready=%.3f "
          "nitros_wrap=%.3f "
          "nitros_publish=%.3fms/frame",
          static_cast<double>(
              diag_frames) / seconds,
          queue_depth,
          DepthSlotCount,
          ms_per_frame(diag_ready_wait),
          ms_per_frame(diag_nitros_wrap),
          ms_per_frame(diag_nitros_publish));

      diag_frames = 0;
      diag_ready_wait = {};
      diag_nitros_wrap = {};
      diag_nitros_publish = {};
      diag_start = now;
    }
  }
}

}  // namespace parallax::ros
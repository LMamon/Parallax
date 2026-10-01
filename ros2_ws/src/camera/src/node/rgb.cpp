#include <parallax/camera/stereo_node.hpp>

namespace parallax::ros {

void StereoNode::rgbPublishLoop() {
  using Clock = std::chrono::steady_clock;

  auto diag_start = Clock::now();
  std::uint64_t diag_frames = 0;

  Clock::duration diag_ready_wait{};
  Clock::duration diag_nitros_wrap{};
  Clock::duration diag_left_publish{};
  Clock::duration diag_right_publish{};
  Clock::duration diag_info_publish{};

  while (rclcpp::ok()) {
    RgbPublication publication;

    {
      std::unique_lock<std::mutex> lock(rgb_mutex_);

      rgb_cv_.wait(lock, [&] {
        return !running_.load() ||
               rgb_size_ != 0;
      });

      //
      // Drain anything already queued before exiting. This preserves
      // ownership of in-flight ISP/rectification buffers through their
      // completion during shutdown.
      //
      if (rgb_size_ == 0) {
        if (!running_.load()) {
          return;
        }

        continue;
      }

      publication =
          std::move(rgb_queue_[rgb_head_]);

      rgb_queue_[rgb_head_] =
          RgbPublication{};

      rgb_head_ =
          (rgb_head_ + 1) %
          RgbQueueCapacity;

      --rgb_size_;
    }

    //
    // This is the host readiness boundary that previously lived in
    // computeLoop().
    //
    const auto ready_start = Clock::now();

    const bool rgb_ready =
        context_.waitForHost(publication.ready);

    diag_ready_wait +=
        Clock::now() - ready_start;

    if (!rgb_ready) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "failed waiting for rectified RGB readiness");

      //
      // Do not publish an image whose producer completion failed.
      //
      publication.isp.reset();
      publication.frame.reset();
      continue;
    }

    //
    // Rectification is complete. VPI can no longer be reading the ISP
    // input, so the ISP output slot can safely return to its pool now.
    //
    publication.isp.reset();

    //
    // During shutdown we still perform the readiness wait above so the
    // pooled resources are retired safely, but there is no reason to send
    // new ROS publications.
    //
    if (!running_.load()) {
      publication.frame.reset();
      continue;
    }

    const auto nitros_wrap_start = Clock::now();

    try {
      auto left_image =
          makePooledNitrosImage(
              publication.frame,
              publication.frame->rgb.left,
              publication.left,
              nvidia::gxf::VideoFormat::
                  GXF_VIDEO_FORMAT_RGB,
              "RGB",
              3U);

      auto right_image =
          makePooledNitrosImage(
              publication.frame,
              publication.frame->rgb.right,
              publication.right,
              nvidia::gxf::VideoFormat::
                  GXF_VIDEO_FORMAT_RGB,
              "RGB",
              3U);

      const auto nitros_wrap_end = Clock::now();

      const auto left_publish_start = Clock::now();
      left_nitros_pub_->publish(left_image);
      const auto left_publish_end = Clock::now();

      right_nitros_pub_->publish(right_image);
      const auto right_publish_end = Clock::now();

      const auto nitros_publish_end = Clock::now();

      diag_nitros_wrap +=
          nitros_wrap_end -
          nitros_wrap_start;

      diag_left_publish +=
        left_publish_end -
        left_publish_start;

      diag_right_publish +=
        right_publish_end -
        left_publish_end;

    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "NITROS spatial publication failed: %s",
          e.what());

      continue;
    }

    //
    // CameraInfo stays on the RGB publication branch so image and
    // calibration metadata retain the same timestamp.
    //
    left_info_.header.stamp =
        publication.left.stamp;

    right_info_.header.stamp =
        publication.right.stamp;

    spatial_left_info_.header.stamp =
        publication.left.stamp;

    spatial_right_info_.header.stamp =
        publication.right.stamp;

    const auto info_start = Clock::now();

    left_info_pub_->publish(left_info_);
    right_info_pub_->publish(right_info_);

    spatial_left_info_pub_->publish(
        spatial_left_info_);

    spatial_right_info_pub_->publish(
        spatial_right_info_);

    diag_info_publish +=
        Clock::now() - info_start;

    ++diag_frames;

    const auto diag_now = Clock::now();

    if (diagnostics_ &&
        diag_now - diag_start >=
            std::chrono::seconds(5)) {

      const double seconds =
          std::chrono::duration<double>(
              diag_now - diag_start).count();

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
            rgb_mutex_);

        queue_depth = rgb_size_;
      }

      RCLCPP_INFO(
          get_logger(),
          "rgb_profile rate=%.2fHz "
          "queue=%zu/%zu "
          "ready=%.3f "
          "nitros_wrap=%.3f "
					"left_pub=%.3f "
					"right_pub=%.3f "
          "info_pub=%.3fms/frame",
          static_cast<double>(
              diag_frames) / seconds,
          queue_depth,
          RgbQueueCapacity,
          ms_per_frame(diag_ready_wait),
          ms_per_frame(diag_nitros_wrap),
					ms_per_frame(diag_left_publish),
					ms_per_frame(diag_right_publish),
          ms_per_frame(diag_info_publish));

      diag_frames = 0;

      diag_ready_wait = {};
      diag_nitros_wrap = {};
			diag_left_publish = {};
			diag_right_publish = {};
      diag_info_publish = {};

      diag_start = diag_now;
    }
  }
}

}  // namespace parallax::ros
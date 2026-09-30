#include <parallax/camera/stereo_node.hpp>

namespace parallax::ros {

void StereoNode::computeLoop() {
  using Clock = std::chrono::steady_clock;

  std::uint64_t last_raw = 0;
  auto diag_start = Clock::now();
  std::uint64_t diag_frames = 0;
  std::uint64_t diag_skipped = 0;
  std::uint64_t diag_rgb_dropped = 0;

  Clock::duration diag_compute{};
  Clock::duration diag_wait{};
  Clock::duration diag_isp_acquire{};
  Clock::duration diag_isp_submit{};
  Clock::duration diag_raw_event{};
  Clock::duration diag_raw_release{};
  Clock::duration diag_isp_completion{};
  Clock::duration diag_isp_dependency{};
  Clock::duration diag_rect_acquire{};
  Clock::duration diag_rect_submit{};
  Clock::duration diag_rect_complete{};
  Clock::duration diag_stereo_handoff{};
  Clock::duration diag_rgb_handoff{};

  while (running_.load() && rclcpp::ok()) {
    const auto wait_start = Clock::now();

    RawLease raw;
    {
      std::unique_lock<std::mutex> lock(raw_mutex_);

      raw_cv_.wait(lock, [&] {
        return !running_.load() || raw_pending_.valid;
      });

      if (!running_.load()) {
        return;
      }

      raw = raw_pending_;
      raw_pending_.valid = false;

      if (last_raw != 0 && raw.sequence > last_raw + 1) {
        diag_skipped += raw.sequence - last_raw - 1;
      }

      last_raw = raw.sequence;
    }

    const auto wait_end = Clock::now();
    diag_wait += wait_end - wait_start;

    // Compute timing intentionally begins after the mailbox wait.
    const auto compute_start = wait_end;

    auto release_raw = [&] {
      if (!raw.valid) {
        return true;
      }

      const bool released = camera_->release(raw.frame);
      raw.valid = false;
      return released;
    };

    const auto isp_acquire_start = Clock::now();

    auto isp_output = isp_.acquireOutput();

    diag_isp_acquire +=
        Clock::now() - isp_acquire_start;

    if (!isp_output) {
      (void)release_raw();
      continue;
    }

    const auto isp_submit_start = Clock::now();

    const bool isp_submitted =
        isp_.processMapped(
            raw.frame,
            *isp_output,
            raw_consumed_event_);

    diag_isp_submit +=
        Clock::now() - isp_submit_start;

    if (!isp_submitted) {
      (void)isp_.synchronize();
      (void)release_raw();

      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "mapped ISP processing failed");

      continue;
    }

    const auto raw_event_start = Clock::now();

    const auto raw_event_status =
        cudaEventSynchronize(raw_consumed_event_);

    diag_raw_event +=
        Clock::now() - raw_event_start;

    const auto raw_release_start = Clock::now();

    const bool raw_released =
        raw_event_status == cudaSuccess &&
        release_raw();

    diag_raw_release +=
        Clock::now() - raw_release_start;

    if (!raw_released) {
      RCLCPP_ERROR(
          get_logger(),
          "failed to recycle mapped camera buffer");

      running_.store(false);
      raw_cv_.notify_all();
      stereo_cv_.notify_all();
      rgb_cv_.notify_all();
      return;
    }

    const auto isp_completion_start = Clock::now();

    auto isp_ready =
        context_.recordCudaCompletion(isp_.stream());

    diag_isp_completion +=
        Clock::now() - isp_completion_start;

    if (!isp_ready.valid()) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "failed to record ISP CUDA completion");

      (void)isp_.synchronize();
      continue;
    }

    auto& preprocess = context_.preprocessLane();

    const auto isp_dependency_start = Clock::now();

    const bool isp_dependency_established =
        context_.waitFor(isp_ready, preprocess);

    diag_isp_dependency +=
        Clock::now() - isp_dependency_start;

    if (!isp_dependency_established) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "failed to establish ISP-to-rectifier CUDA dependency");

      (void)isp_.synchronize();
      continue;
    }

    const auto rect_acquire_start = Clock::now();

    auto rectified =
        rectifier_.acquireOutput();

    diag_rect_acquire +=
        Clock::now() - rect_acquire_start;

    if (!rectified) {
      continue;
    }

    const auto rect_submit_start = Clock::now();

    const bool rect_submitted =
        rectifier_.process(
            isp_output->rgb,
            isp_output->gray,
            *rectified,
            preprocess.handle());

    diag_rect_submit +=
        Clock::now() - rect_submit_start;

    if (!rect_submitted) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "rectification failed");

      continue;
    }

    const auto rect_complete_start = Clock::now();

    auto rect_ready =
        context_.recordVpiCompletion(
            preprocess.handle());

    diag_rect_complete +=
        Clock::now() - rect_complete_start;

    if (!rect_ready.valid()) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "failed to record rectification completion");

      (void)preprocess.synchronize();
      continue;
    }

    std_msgs::msg::Header left_header;
    left_header.stamp =
        wallStamp(raw.wall_timestamp);
    left_header.frame_id = kLeftFrame;

    std_msgs::msg::Header right_header;
    right_header.stamp = left_header.stamp;
    right_header.frame_id = kRightFrame;

    //
    // Stereo branch.
    //
    // Keep this dependency exactly as before. The stereo worker receives
    // the same rectified output and the same rectification completion.
    //
    const auto stereo_handoff_start = Clock::now();

    {
      std::lock_guard<std::mutex> lock(stereo_mutex_);

      stereo_pending_ = rectified;
      stereo_pending_stamp_ = left_header.stamp;
      stereo_pending_ready_ = rect_ready;
    }

    stereo_cv_.notify_one();

    diag_stereo_handoff +=
        Clock::now() - stereo_handoff_start;

    //
    // RGB branch.
    //
    // The RGB publication owns isp_output until rect_ready completes.
    // That is required because rectification may still be reading from
    // the ISP output after this function has submitted the VPI work.
    //
    const auto rgb_handoff_start = Clock::now();

    RgbPublication publication{
        .isp = isp_output,
        .frame = rectified,
        .ready = rect_ready,
        .left = left_header,
        .right = right_header,
    };

    bool rgb_enqueued = false;

    {
      std::lock_guard<std::mutex> lock(rgb_mutex_);

      if (rgb_size_ < RgbQueueCapacity) {
        rgb_queue_[rgb_tail_] =
            std::move(publication);

        rgb_tail_ =
            (rgb_tail_ + 1) %
            RgbQueueCapacity;

        ++rgb_size_;
        rgb_enqueued = true;
      }
    }

    if (rgb_enqueued) {
      rgb_cv_.notify_one();
    } else {
      //
      // Never overwrite an occupied ring slot.
      //
      // publication still owns isp_output here. Because the queue is
      // overloaded, wait until rectification is finished before allowing
      // that ISP slot to return to its pool.
      //
      const bool retired =
          context_.waitForHost(publication.ready);

      if (!retired) {
        RCLCPP_ERROR_THROTTLE(
            get_logger(),
            *get_clock(),
            2000,
            "failed retiring dropped RGB publication");
      }

      publication.isp.reset();
      publication.frame.reset();

      ++diag_rgb_dropped;
    }

    diag_rgb_handoff +=
        Clock::now() - rgb_handoff_start;

    //
    // There is intentionally no RGB readiness wait or NITROS publication
    // below this point. The RGB worker owns that work now.
    //

    diag_compute +=
        Clock::now() - compute_start;

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

      RCLCPP_INFO(
          get_logger(),
          "compute_profile rate=%.2fHz "
          "skipped=%llu rgb_dropped=%llu "
          "wait=%.3f "
          "isp_acq=%.3f isp_submit=%.3f "
          "raw_event=%.3f raw_release=%.3f "
          "isp_complete=%.3f isp_dep=%.3f "
          "rect_acq=%.3f rect_submit=%.3f "
          "rect_complete=%.3f "
          "stereo_handoff=%.3f rgb_handoff=%.3f "
          "compute=%.3fms/frame",
          static_cast<double>(
              diag_frames) / seconds,
          static_cast<unsigned long long>(
              diag_skipped),
          static_cast<unsigned long long>(
              diag_rgb_dropped),
          ms_per_frame(diag_wait),
          ms_per_frame(diag_isp_acquire),
          ms_per_frame(diag_isp_submit),
          ms_per_frame(diag_raw_event),
          ms_per_frame(diag_raw_release),
          ms_per_frame(diag_isp_completion),
          ms_per_frame(diag_isp_dependency),
          ms_per_frame(diag_rect_acquire),
          ms_per_frame(diag_rect_submit),
          ms_per_frame(diag_rect_complete),
          ms_per_frame(diag_stereo_handoff),
          ms_per_frame(diag_rgb_handoff),
          ms_per_frame(diag_compute));

      diag_frames = 0;
      diag_skipped = 0;
      diag_rgb_dropped = 0;

      diag_compute = {};
      diag_wait = {};
      diag_isp_acquire = {};
      diag_isp_submit = {};
      diag_raw_event = {};
      diag_raw_release = {};
      diag_isp_completion = {};
      diag_isp_dependency = {};
      diag_rect_acquire = {};
      diag_rect_submit = {};
      diag_rect_complete = {};
      diag_stereo_handoff = {};
      diag_rgb_handoff = {};

      diag_start = diag_now;
    }
  }
}

}  // namespace parallax::ros
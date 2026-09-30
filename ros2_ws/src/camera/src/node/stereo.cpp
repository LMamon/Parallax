#include <parallax/camera/stereo_node.hpp>
#include <parallax/cuda/depth.cuh>
#include <parallax/cuda/gray_resize.cuh>
#include <parallax/stereo/depth_policy.hpp>

namespace parallax::ros {

void StereoNode::stereoLoop() {
  using Clock = std::chrono::steady_clock;

  std::uint64_t completed = 0;
  std::uint64_t superseded = 0;
  auto diag_start = Clock::now();

  Clock::duration diag_total{};
  Clock::duration diag_wait{};
  Clock::duration diag_dependency{};
  Clock::duration diag_downsample{};
  Clock::duration diag_acquire{};
  Clock::duration diag_matcher{};
  Clock::duration diag_depth{};
  Clock::duration diag_completion{};

  while (running_.load() && rclcpp::ok()) {
    const auto wait_start = Clock::now();

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

    const auto wait_end = Clock::now();
    diag_wait += wait_end - wait_start;

    // Stereo processing time intentionally begins after the mailbox wait.
    const auto total_start = wait_end;

    auto& lane = context_.stereoLane();

    const auto dependency_start = Clock::now();

    const bool dependency_established =
        context_.waitFor(rect_ready, lane);

    diag_dependency +=
        Clock::now() - dependency_start;

    if (!dependency_established) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "failed to establish rectifier-to-stereo dependency");

      continue;
    }

    auto spatial = spatial_gray_pool_.acquire();

    if (!spatial) {
        ++superseded;
        continue;
    }

    const auto downsample_start = Clock::now();

    const bool downsampled =
        parallax::cuda::downsampleGray2x(
            rectified->gray.left,
            spatial->frame.left,
            lane.cudaHandle()) &&
        parallax::cuda::downsampleGray2x(
            rectified->gray.right,
            spatial->frame.right,
            lane.cudaHandle());

    diag_downsample +=
        Clock::now() - downsample_start;

    if (!downsampled) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "gray spatial downsample failed");

      continue;
    }

    const auto acquire_start = Clock::now();

    auto match =
        matcher_.acquireOutput(context_);

    auto depth =
        depth_pool_.acquire();

    diag_acquire +=
        Clock::now() - acquire_start;

    if (!match || !depth) {
      ++superseded;
      continue;
    }

    const auto matcher_start = Clock::now();

    const bool matched =
        matcher_.process(
            spatial->frame,
            *match,
            lane.handle());

    diag_matcher +=
        Clock::now() - matcher_start;

    if (!matched) {
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

    const auto depth_start = Clock::now();

    const bool depth_converted =
        parallax::cuda::disparityToDepth(
            match->output.disparity,
            depth->depth,
            fx,
            baseline,
            parallax::isp::StereoMatchFrame::DisparityScale,
            parallax::stereo::MinUsefulDepthM,
            parallax::stereo::MaxUsefulDepthM,
            lane.cudaHandle());

    diag_depth +=
        Clock::now() - depth_start;

    if (!depth_converted) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "disparity-to-depth failed");

      continue;
    }

    //
    const auto completion_start = Clock::now();

    auto depth_ready =
        context_.recordVpiCompletion(
            lane.handle());

    diag_completion +=
        Clock::now() - completion_start;

    if (!depth_ready.valid()) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "failed to record stereo depth completion");

      //
      // We cannot safely release spatial/match/depth while submitted
      // work may still reference them.
      //
      (void)lane.synchronize();
      continue;
    }

    std_msgs::msg::Header header;
    header.stamp = stamp;
    header.frame_id = kLeftFrame;

    DepthPublication publication{
        .spatial = std::move(spatial),
        .match = std::move(match),
        .depth = std::move(depth),
        .ready = depth_ready,
        .header = std::move(header),
    };

    bool depth_enqueued = false;

    {
      std::lock_guard<std::mutex> lock(
          depth_mutex_);

      if (depth_size_ < DepthSlotCount) {
        depth_queue_[depth_tail_] =
            std::move(publication);

        depth_tail_ =
            (depth_tail_ + 1) %
            DepthSlotCount;

        ++depth_size_;
        depth_enqueued = true;
      }
    }

    if (depth_enqueued) {
      depth_cv_.notify_one();
    } else {
      //
      // Never release a generation whose GPU work is still in flight.
      // This should be exceptional because the resource pools and queue
      // have the same bounded depth.
      //
      const bool retired =
          context_.waitForHost(
              publication.ready);

      if (!retired) {
        RCLCPP_ERROR_THROTTLE(
            get_logger(),
            *get_clock(),
            2000,
            "failed retiring dropped depth generation");
      }

      publication.spatial.reset();
      publication.match.reset();
      publication.depth.reset();

      ++superseded;
      continue;
    }

    diag_total +=
        Clock::now() - total_start;

    ++completed;

    const auto now = Clock::now();

    if (diagnostics_ &&
        now - diag_start >=
            std::chrono::seconds(5)) {

      const double seconds =
          std::chrono::duration<double>(
              now - diag_start).count();

      const auto ms_per_frame =
          [completed](
              Clock::duration duration) -> double {

        if (completed == 0) {
          return 0.0;
        }

        return
            std::chrono::duration<
                double,
                std::milli>(duration).count() /
            static_cast<double>(completed);
      };

      RCLCPP_INFO(
          get_logger(),
          "stereo_profile rate=%.2fHz "
          "superseded=%llu "
          "wait=%.3f dep=%.3f "
          "downsample=%.3f acquire=%.3f "
          "matcher=%.3f depth=%.3f "
          "completion=%.3f "
          "total=%.3fms/frame "
          "path=Y8->CUDA_SGM->depth",
          static_cast<double>(
              completed) / seconds,
          static_cast<unsigned long long>(
              superseded),
          ms_per_frame(diag_wait),
          ms_per_frame(diag_dependency),
          ms_per_frame(diag_downsample),
          ms_per_frame(diag_acquire),
          ms_per_frame(diag_matcher),
          ms_per_frame(diag_depth),
          ms_per_frame(diag_completion),
          ms_per_frame(diag_total));

      completed = 0;
      superseded = 0;

      diag_total = {};
      diag_wait = {};
      diag_dependency = {};
      diag_downsample = {};
      diag_acquire = {};
      diag_matcher = {};
      diag_depth = {};
      diag_completion = {};

      diag_start = now;
    }
  }
}

}  // namespace parallax::ros
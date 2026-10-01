#include <parallax/camera/stereo_node.hpp>

namespace parallax::ros {

    void StereoNode::computeLoop() {
        while (running_.load() && rclcpp::ok()) {
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
            }

            auto release_raw = [&] {
            if (!raw.valid) {
                return true;
            }

            const bool released = camera_->release(raw.frame);
            raw.valid = false;
            return released;
            };

            auto isp_output = isp_.acquireOutput();

            if (!isp_output) {
            (void)release_raw();
            continue;
            }

            const bool isp_submitted =
                isp_.processMapped(
                    raw.frame,
                    *isp_output,
                    raw_consumed_event_);

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

            const auto raw_event_status =
                cudaEventSynchronize(raw_consumed_event_);

            const bool raw_released =
                raw_event_status == cudaSuccess &&
                release_raw();

            if (!raw_released) {
            RCLCPP_ERROR(
                get_logger(),
                "failed to recycle mapped camera buffer");

            running_.store(false);
            raw_cv_.notify_all();
            rgb_cv_.notify_all();
            return;
            }

            auto isp_ready =
                context_.recordCudaCompletion(isp_.stream());

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

            const bool isp_dependency_established =
                context_.waitFor(isp_ready, preprocess);

            if (!isp_dependency_established) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "failed to establish ISP-to-rectifier CUDA dependency");

            (void)isp_.synchronize();
            continue;
            }

            auto rectified =
                rectifier_.acquireOutput();

            if (!rectified) {
            continue;
            }

            const bool rect_submitted =
                rectifier_.process(
                    isp_output->rgb,
                    isp_output->gray,
                    *rectified,
                    preprocess.handle());

            if (!rect_submitted) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "rectification failed");

            continue;
            }

            auto rect_ready =
                context_.recordVpiCompletion(
                    preprocess.handle());

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
            // RGB branch.
            //
            // The RGB publication owns isp_output until rect_ready completes.
            // That is required because rectification may still be reading from
            // the ISP output after this function has submitted the VPI work.
            //
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
            }

            //
            // There is intentionally no RGB readiness wait or NITROS publication
            // below this point. The RGB worker owns that work now.
            //
        }
    }

}  // namespace parallax::ros
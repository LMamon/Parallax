#include <parallax/camera/stereo_node.hpp>

namespace parallax::ros {

    void StereoNode::rgbPublishLoop() {
        using Clock = std::chrono::steady_clock;

        while (rclcpp::ok()) {
            RgbPublication publication;

            {
                std::unique_lock<std::mutex> lock(rgb_mutex_);

                rgb_cv_.wait(lock, [&] {
                    return !running_.load() ||rgb_size_ != 0;
                });


                // Drain anything already queued before exiting. This preserves
                // ownership of in-flight ISP/rectification buffers through their
                // completion during shutdown.
                if (rgb_size_ == 0) {
                    if (!running_.load()) {
                        return;
                    }

                    continue;
                }

                publication = std::move(rgb_queue_[rgb_head_]);

                rgb_queue_[rgb_head_] = RgbPublication{};

                rgb_head_ = (rgb_head_ + 1) % RgbQueueCapacity;
                --rgb_size_;
            }

            // This is the host readiness boundary that previously lived in
            // computeLoop().
            const bool rgb_ready = context_.waitForHost(publication.ready);

            if (!rgb_ready) {
                RCLCPP_ERROR_THROTTLE(get_logger(),
                                    *get_clock(),
                                    2000,
                                    "failed waiting for rectified RGB readiness");

                // Do not publish an image whose producer completion failed.
                publication.isp.reset();
                publication.frame.reset();
                continue;
            }

            // Rectification is complete. VPI can no longer be reading the ISP
            // input, so the ISP output slot can safely return to its pool now.
            publication.isp.reset();

            // During shutdown we still perform the readiness wait above so the
            // pooled resources are retired safely, but there is no reason to send
            // new ROS publications.
            if (!running_.load()) {
                publication.frame.reset();
                continue;
            }

            try {
                auto left_image = makePooledNitrosImage(publication.frame,
                                                        publication.frame->rgb.left,
                                                        publication.left,
                                                        nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB,
                                                        "RGB",
                                                        3U);

                auto right_image = makePooledNitrosImage(publication.frame,
                                                        publication.frame->rgb.right,
                                                        publication.right,
                                                        nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB,
                                                        "RGB",
                                                        3U);

                left_nitros_pub_->publish(left_image);
                right_nitros_pub_->publish(right_image);
            } catch (const std::exception& e) {
                RCLCPP_ERROR_THROTTLE(get_logger(),
                                    *get_clock(),
                                    2000,
                                    "NITROS spatial publication failed: %s",
                                    e.what());

                continue;
            }

            // CameraInfo stays on the RGB publication branch so image and
            // calibration metadata retain the same timestamp.
            left_info_.header.stamp = publication.left.stamp;
            right_info_.header.stamp = publication.right.stamp;

            spatial_left_info_.header.stamp = publication.left.stamp;
            spatial_right_info_.header.stamp = publication.right.stamp;

            left_info_pub_->publish(left_info_);
            right_info_pub_->publish(right_info_);

            spatial_left_info_pub_->publish(spatial_left_info_);
            spatial_right_info_pub_->publish(spatial_right_info_);

            // Visualization is intentionally rate-limited and independent of
            // the full-rate NITROS compute path.
            const auto preview_now = Clock::now();

            if (preview_now >= preview_next_) {
                preview_next_ = preview_now + std::chrono::microseconds(1000000 / preview_fps_);

                {
                    std::lock_guard<std::mutex> lock(preview_mutex_);

                    preview_pending_.frame = publication.frame;
                    preview_pending_.left = publication.left;
                    preview_pending_.right = publication.right;
                    preview_pending_.valid = true;
                }

                preview_cv_.notify_one();
            }
        }
    }

}
#include <parallax/camera/stereo_node.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace parallax::ros {

    void StereoNode::previewLoop() {
        std::vector<std::uint8_t> left_host_rgb;
        std::vector<std::uint8_t> right_host_rgb;

        cv::Mat left_preview;
        cv::Mat right_preview;

        while (rclcpp::ok()) {
            PreviewPublication publication;
            {
                std::unique_lock<std::mutex> lock(preview_mutex_);
                preview_cv_.wait(lock, [&] {
                    return !running_.load() || preview_pending_.valid;
                });
                if (!preview_pending_.valid) {
                    if (!running_.load()) return;
                    continue;
                }

                publication = std::move(preview_pending_);
                preview_pending_ = PreviewPublication{};
            }

            if (!running_.load() || !publication.frame) continue;

            const auto& left_gpu = publication.frame->rgb.left;
            const auto& right_gpu = publication.frame->rgb.right;

            const std::size_t left_row_bytes = static_cast<std::size_t>(left_gpu.width()) * 3U;
            const std::size_t right_row_bytes = static_cast<std::size_t>(right_gpu.width()) * 3U;

            left_host_rgb.resize(left_row_bytes * static_cast<std::size_t>(left_gpu.height()));
            right_host_rgb.resize(right_row_bytes * static_cast<std::size_t>(right_gpu.height()));

            const auto left_copy = cudaMemcpy2DAsync(
                left_host_rgb.data(), left_row_bytes,
                left_gpu.data(), left_gpu.pitch(),
                left_row_bytes, left_gpu.height(),
                cudaMemcpyDeviceToHost, preview_stream_);

            const auto right_copy = cudaMemcpy2DAsync(
                right_host_rgb.data(), right_row_bytes,
                right_gpu.data(), right_gpu.pitch(),
                right_row_bytes, right_gpu.height(),
                cudaMemcpyDeviceToHost, preview_stream_);

            const auto copy_sync = cudaStreamSynchronize(preview_stream_);
            if (left_copy != cudaSuccess || right_copy != cudaSuccess || copy_sync != cudaSuccess) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                     "stereo preview GPU-to-host copy failed");
                continue;
            }

            cv::Mat left_full(static_cast<int>(left_gpu.height()),
                              static_cast<int>(left_gpu.width()),
                              CV_8UC3,
                              left_host_rgb.data(),
                              left_row_bytes);

            cv::Mat right_full(static_cast<int>(right_gpu.height()),
                               static_cast<int>(right_gpu.width()),
                               CV_8UC3,
                               right_host_rgb.data(),
                               right_row_bytes);

            cv::resize(left_full, left_preview,
                       cv::Size(static_cast<int>(left_gpu.width() / 2U),
                                static_cast<int>(left_gpu.height() / 2U)),
                       0.0,
                       0.0,
                       cv::INTER_AREA);

            cv::resize(right_full, right_preview,
                       cv::Size(static_cast<int>(right_gpu.width() / 2U),
                                static_cast<int>(right_gpu.height() / 2U)),
                       0.0,
                       0.0,
                       cv::INTER_AREA);

            cv::Mat left_bgr;
            cv::Mat right_bgr;
            cv::cvtColor(left_preview, left_bgr, cv::COLOR_RGB2BGR);
            cv::cvtColor(right_preview, right_bgr, cv::COLOR_RGB2BGR);

            std::vector<std::uint8_t> left_jpeg;
            std::vector<std::uint8_t> right_jpeg;

            const std::vector<int> jpeg_params{cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};

            const bool left_encoded = cv::imencode(".jpg", left_bgr, left_jpeg, jpeg_params);
            const bool right_encoded = cv::imencode(".jpg", right_bgr, right_jpeg, jpeg_params);

            if (!left_encoded || !right_encoded) {
                continue;
            }

            sensor_msgs::msg::CompressedImage left_msg;
            left_msg.header = publication.left;
            left_msg.format = "jpeg";
            left_msg.data = std::move(left_jpeg);

            sensor_msgs::msg::CompressedImage right_msg;
            right_msg.header = publication.right;
            right_msg.format = "jpeg";
            right_msg.data = std::move(right_jpeg);

            preview_left_pub_->publish(std::move(left_msg));
            preview_right_pub_->publish(std::move(right_msg));
        }
    }

}

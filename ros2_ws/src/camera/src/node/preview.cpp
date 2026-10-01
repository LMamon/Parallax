#include <parallax/camera/stereo_node.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace parallax::ros {

void StereoNode::previewLoop() {
  std::vector<std::uint8_t> host_rgb;
  cv::Mat preview;

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

    const auto& gpu = publication.frame->rgb.left;
    const std::size_t row_bytes =
        static_cast<std::size_t>(gpu.width()) * 3U;
    host_rgb.resize(
        row_bytes * static_cast<std::size_t>(gpu.height()));

    if (cudaMemcpy2DAsync(
            host_rgb.data(), row_bytes,
            gpu.data(), gpu.pitch(),
            row_bytes, gpu.height(),
            cudaMemcpyDeviceToHost,
            preview_stream_) != cudaSuccess ||
        cudaStreamSynchronize(preview_stream_) != cudaSuccess) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "preview GPU-to-host copy failed");
      continue;
    }

    cv::Mat full(
        static_cast<int>(gpu.height()),
        static_cast<int>(gpu.width()),
        CV_8UC3, host_rgb.data(), row_bytes);

    cv::resize(
        full, preview,
        cv::Size(
            static_cast<int>(gpu.width() / 2U),
            static_cast<int>(gpu.height() / 2U)),
        0.0, 0.0, cv::INTER_AREA);

    std::vector<std::uint8_t> jpeg;
    if (!cv::imencode(
            ".jpg", preview, jpeg,
            {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_})) {
      continue;
    }

    sensor_msgs::msg::CompressedImage msg;
    msg.header = publication.header;
    msg.format = "jpeg";
    msg.data = std::move(jpeg);
    preview_left_pub_->publish(std::move(msg));
  }
}

}  // namespace parallax::ros

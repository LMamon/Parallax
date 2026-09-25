#include <parallax/camera/arducam_controls.hpp>
#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/camera_producer.hpp>
#include <parallax/camera/stereo_camera.hpp>
#include <parallax/core/execution_context.hpp>
#include <parallax/core/product_id.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/core/producer.hpp>
#include <parallax/isp/auto_control.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/isp/isp.hpp>
#include <parallax/isp/isp_config.hpp>
#include <parallax/isp/isp_producer.hpp>
#include <parallax/stereo/calibration.hpp>
#include <parallax/stereo/rectification.hpp>
#include <parallax/stereo/rectification_producer.hpp>

#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using parallax::core::ProductId;
using parallax::core::SubmitResult;

constexpr char kLeftFrame[] = "left_camera_optical_frame";
constexpr char kRightFrame[] = "right_camera_optical_frame";

sensor_msgs::msg::CameraInfo makeRectifiedInfo(
    const parallax::stereo::StereoCalibration& calibration,
    const std::array<double, 12>& projection,
    const char* frame_id) {
  sensor_msgs::msg::CameraInfo info;
  const auto& meta = calibration.metadata();

  info.width = meta.image_width;
  info.height = meta.image_height;
  info.header.frame_id = frame_id;
  info.distortion_model = "plumb_bob";
  info.d.assign(5, 0.0);
  info.k = {
      projection[0], projection[1], projection[2],
      projection[4], projection[5], projection[6],
      projection[8], projection[9], projection[10],
  };
  info.r = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
  };
  info.p = projection;
  return info;
}

rclcpp::Time productStamp(const parallax::core::ProductMetadata& metadata) {
  if (!metadata.wall_timestamp_valid) {
    return rclcpp::Clock(RCL_SYSTEM_TIME).now();
  }

  return rclcpp::Time(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          metadata.wall_timestamp.time_since_epoch()).count(),
      RCL_SYSTEM_TIME);
}

class StereoNode final : public rclcpp::Node {
 public:
  StereoNode() : Node("stereo_camera") {
    const auto camera_config_path =
        declare_parameter<std::string>("camera_config", "");
    const auto isp_config_path =
        declare_parameter<std::string>("isp_config", "");
    const auto calibration_dir =
        declare_parameter<std::string>("calibration_dir", "");
    preview_fps_ = declare_parameter<int>("preview_fps", 20);
    jpeg_quality_ = declare_parameter<int>("jpeg_quality", 85);

    if (preview_fps_ < 1 || jpeg_quality_ < 1 || jpeg_quality_ > 100) {
      throw std::runtime_error("invalid preview configuration");
    }

    if (camera_config_path.empty() ||
        !camera_config_.loadFromFile(camera_config_path)) {
      throw std::runtime_error("failed to load stereo camera config");
    }
    if (isp_config_path.empty() ||
        !isp_config_.loadFromFile(isp_config_path)) {
      throw std::runtime_error("failed to load ISP config");
    }
    if (calibration_dir.empty() || !calibration_.load(calibration_dir)) {
      throw std::runtime_error("failed to load stereo calibration");
    }

    if (!context_.initialize()) {
      throw std::runtime_error("failed to initialize execution context");
    }

    camera_ =
        std::make_unique<parallax::camera::StereoCamera>(camera_config_);
    if (!camera_->initialize()) {
      throw std::runtime_error("failed to initialize AR0234 stereo camera");
    }

    if (!isp_.initialize(camera_config_, isp_config_)) {
      throw std::runtime_error("failed to initialize ISP");
    }

    // Match the proven graph path: rectification consumes ISP-owned device
    // storage on the shared preprocess lane. No host synchronization here.
    auto& preprocess = context_.preprocessLane();
    if (!rectifier_.initialize(
            calibration_, isp_.rgb(), isp_.gray(), preprocess.handle())) {
      throw std::runtime_error("failed to initialize stereo rectifier");
    }

    initializeAutoControl();

    camera_producer_ =
        std::make_unique<parallax::camera::CameraProducer>(
            *camera_, context_.products());
    isp_producer_ =
        std::make_unique<parallax::isp::IspProducer>(
            isp_, context_.products());
    rectification_producer_ =
        std::make_unique<parallax::stereo::RectificationProducer>(
            rectifier_, calibration_, context_.products());

    const auto qos = rclcpp::SensorDataQoS().keep_last(1);

    left_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "/stereo/left/image_rect", qos);
    right_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "/stereo/right/image_rect", qos);
    left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "/stereo/left/camera_info", qos);
    right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "/stereo/right/camera_info", qos);

    // Foxglove-facing preview. This is intentionally not part of the
    // acquisition/compute cadence.
    color_preview_pub_ =
        create_publisher<sensor_msgs::msg::CompressedImage>(
            "/stereo/left/image_rect_color/compressed", qos);

    left_info_ =
        makeRectifiedInfo(calibration_, calibration_.P1(), kLeftFrame);
    right_info_ =
        makeRectifiedInfo(calibration_, calibration_.P2(), kRightFrame);

    const auto pixels =
        static_cast<std::size_t>(calibration_.metadata().image_width) *
        calibration_.metadata().image_height;
    left_gray_.resize(pixels);
    right_gray_.resize(pixels);
    left_rgb_.resize(pixels * 3U);

    running_.store(true);
    capture_thread_ = std::thread(&StereoNode::captureLoop, this);
    publication_thread_ = std::thread(&StereoNode::publicationLoop, this);
    auto_control_thread_ = std::thread(&StereoNode::autoControlLoop, this);

    RCLCPP_INFO(
        get_logger(),
        "AR0234 graph boundary ready: capture -> async ISP -> async VPI; "
        "rectified mono ROS + %dHz compressed RGB preview; AE=%s AWB=%s",
        preview_fps_,
        isp_config_.auto_exposure.enable ? "on" : "off",
        isp_config_.auto_white_balance.enable ? "on" : "off");
  }

  ~StereoNode() override {
    running_.store(false);

    if (capture_thread_.joinable()) capture_thread_.join();
    if (publication_thread_.joinable()) publication_thread_.join();
    if (auto_control_thread_.joinable()) auto_control_thread_.join();

    // Drain accelerator work before releasing product generations/storage.
    (void)context_.drain();
    context_.products().clear();

    rectification_producer_.reset();
    isp_producer_.reset();
    camera_producer_.reset();
    auto_controller_.reset();

    rectifier_.shutdown();
    isp_.shutdown();

    if (camera_) {
      camera_->shutdown();
      camera_.reset();
    }

    context_.shutdown();
  }

 private:
  void initializeAutoControl() {
    if (!isp_config_.auto_exposure.enable &&
        !isp_config_.auto_white_balance.enable) {
      return;
    }

    parallax::camera::ControlRange exposure_range{};
    parallax::camera::ControlRange gain_range{};

    if (isp_config_.auto_exposure.enable) {
      if (!camera_->getControlRange(
              parallax::camera::controls::Exposure, exposure_range) ||
          !camera_->getControlRange(
              parallax::camera::controls::AnalogGain, gain_range)) {
        throw std::runtime_error(
            "failed to query exposure/gain control ranges");
      }
    } else {
      exposure_range = {
          static_cast<std::int32_t>(camera_config_.exposure),
          static_cast<std::int32_t>(camera_config_.exposure),
          1,
          static_cast<std::int32_t>(camera_config_.exposure),
          true,
      };
      gain_range = {
          static_cast<std::int32_t>(camera_config_.analogue_gain),
          static_cast<std::int32_t>(camera_config_.analogue_gain),
          1,
          static_cast<std::int32_t>(camera_config_.analogue_gain),
          true,
      };
    }

    auto_controller_ = std::make_unique<parallax::isp::AutoController>(
        isp_config_,
        exposure_range,
        gain_range,
        static_cast<std::int32_t>(camera_config_.exposure),
        static_cast<std::int32_t>(camera_config_.analogue_gain));
  }

  void autoControlLoop() {
    using namespace std::chrono_literals;

    while (running_.load() && rclcpp::ok() && auto_controller_) {
      parallax::isp::IspStatistics statistics{};

      if (isp_.tryGetStatistics(statistics)) {
        const auto update = auto_controller_->update(statistics);

        if (update.gain_changed &&
            !camera_->setControl(
                parallax::camera::controls::AnalogGain,
                update.analogue_gain)) {
          RCLCPP_ERROR(
              get_logger(),
              "automatic gain update failed; disabling auto control");
          return;
        }

        if (update.exposure_changed &&
            !camera_->setControl(
                parallax::camera::controls::Exposure,
                update.exposure)) {
          RCLCPP_ERROR(
              get_logger(),
              "automatic exposure update failed; disabling auto control");
          return;
        }

        if (update.white_balance_changed) {
          isp_.setWhiteBalance(update.white_balance);
        }
      }

      std::this_thread::sleep_for(5ms);
    }
  }

  void captureLoop() {
    unsigned failures = 0;

    while (running_.load() && rclcpp::ok()) {
      const auto camera_result = camera_producer_->submit(context_);
      if (camera_result != SubmitResult::Submitted) {
        if (++failures >= 10) {
          RCLCPP_ERROR(get_logger(), "camera producer repeatedly failed");
          running_.store(false);
          return;
        }
        continue;
      }

      const auto isp_result = isp_producer_->submit(context_);
      if (isp_result == SubmitResult::Failed) {
        if (++failures >= 10) {
          RCLCPP_ERROR(get_logger(), "ISP producer repeatedly failed");
          running_.store(false);
          return;
        }
        continue;
      }
      if (isp_result == SubmitResult::NoWork) {
        // Fixed pools are intentionally bounded. A slow consumer causes this
        // observation to be superseded rather than growing a frame queue.
        continue;
      }

      const auto rect_result =
          rectification_producer_->submit(context_);
      if (rect_result == SubmitResult::Failed) {
        if (++failures >= 10) {
          RCLCPP_ERROR(
              get_logger(), "rectification producer repeatedly failed");
          running_.store(false);
          return;
        }
        continue;
      }

      if (rect_result == SubmitResult::Submitted) {
        failures = 0;
      }
    }
  }

  bool download(
      const parallax::cuda::CudaBuffer& source,
      std::uint8_t* destination,
      std::size_t host_pitch) {
    return cudaMemcpy2D(
               destination,
               host_pitch,
               source.data(),
               source.pitch(),
               source.rowBytes(),
               source.height(),
               cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  void publishMono(
      const parallax::core::Product<
          parallax::isp::RectifiedStereoGrayFrame>& product) {
    const auto& frame = *product.payload;
    const auto width = frame.width;
    const auto height = frame.height;

    if (!download(frame.left, left_gray_.data(), width) ||
        !download(frame.right, right_gray_.data(), width)) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "rectified mono download failed");
      return;
    }

    const auto stamp = productStamp(product.metadata);

    sensor_msgs::msg::Image left;
    left.header.stamp = stamp;
    left.header.frame_id = kLeftFrame;
    left.height = height;
    left.width = width;
    left.encoding = "mono8";
    left.is_bigendian = false;
    left.step = width;
    left.data = left_gray_;

    sensor_msgs::msg::Image right;
    right.header.stamp = stamp;
    right.header.frame_id = kRightFrame;
    right.height = height;
    right.width = width;
    right.encoding = "mono8";
    right.is_bigendian = false;
    right.step = width;
    right.data = right_gray_;

    left_info_.header.stamp = stamp;
    right_info_.header.stamp = stamp;

    left_image_pub_->publish(std::move(left));
    right_image_pub_->publish(std::move(right));
    left_info_pub_->publish(left_info_);
    right_info_pub_->publish(right_info_);
  }

  void publishColorPreview(
      const parallax::core::Product<
          parallax::isp::RectifiedStereoFrame>& product) {
    const auto now = std::chrono::steady_clock::now();
    const auto interval =
        std::chrono::duration<double>(
            1.0 / static_cast<double>(preview_fps_));

    if (has_preview_publish_ &&
        now - last_preview_publish_ < interval) {
      return;
    }

    const auto& frame = *product.payload;
    const std::size_t pitch =
        static_cast<std::size_t>(frame.width) * 3U;

    if (!download(frame.left, left_rgb_.data(), pitch)) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "rectified RGB preview download failed");
      return;
    }

    cv::Mat rgb(
        static_cast<int>(frame.height),
        static_cast<int>(frame.width),
        CV_8UC3,
        left_rgb_.data(),
        pitch);
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

    std::vector<std::uint8_t> jpeg;
    const std::vector<int> parameters{
        cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
    if (!cv::imencode(".jpg", bgr, jpeg, parameters)) {
      return;
    }

    sensor_msgs::msg::CompressedImage message;
    message.header.stamp = productStamp(product.metadata);
    message.header.frame_id = kLeftFrame;
    message.format = "jpeg";
    message.data = std::move(jpeg);

    color_preview_pub_->publish(std::move(message));
    last_preview_publish_ = now;
    has_preview_publish_ = true;
  }

  void publicationLoop() {
    using namespace std::chrono_literals;

    parallax::core::SourceObservation last_mono{};
    parallax::core::SourceObservation last_preview{};

    while (running_.load() && rclcpp::ok()) {
      // ProductStore is latest-value storage, not a frame queue. If publication
      // is slower than capture, intermediate observations disappear here rather
      // than accumulating latency.
      const auto gray =
          context_.products().latest<
              parallax::isp::RectifiedStereoGrayFrame>(
              ProductId::RectifiedGray);

      if (gray && gray->valid() &&
          gray->metadata.observation != last_mono) {
        if (context_.waitForHost(gray->completion)) {
          publishMono(*gray);
          last_mono = gray->metadata.observation;
        }
      }

      const auto rgb =
          context_.products().latest<
              parallax::isp::RectifiedStereoFrame>(
              ProductId::RectifiedRgb);

      if (rgb && rgb->valid() &&
          rgb->metadata.observation != last_preview) {
        const auto now = std::chrono::steady_clock::now();
        const auto interval =
            std::chrono::duration<double>(
                1.0 / static_cast<double>(preview_fps_));

        if ((!has_preview_publish_ ||
             now - last_preview_publish_ >= interval) &&
            context_.waitForHost(rgb->completion)) {
          publishColorPreview(*rgb);
        }

        // Mark the observation consumed even when preview rate-limited.
        last_preview = rgb->metadata.observation;
      }

      std::this_thread::sleep_for(1ms);
    }
  }

  parallax::camera::CameraConfig camera_config_{};
  parallax::isp::IspConfig isp_config_{};
  parallax::stereo::StereoCalibration calibration_{};

  parallax::core::ExecutionContext context_{};
  std::unique_ptr<parallax::camera::StereoCamera> camera_;
  parallax::isp::ISP isp_{};
  parallax::stereo::StereoRectifier rectifier_{};

  std::unique_ptr<parallax::camera::CameraProducer> camera_producer_;
  std::unique_ptr<parallax::isp::IspProducer> isp_producer_;
  std::unique_ptr<parallax::stereo::RectificationProducer>
      rectification_producer_;
  std::unique_ptr<parallax::isp::AutoController> auto_controller_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr
      color_preview_pub_;

  sensor_msgs::msg::CameraInfo left_info_{};
  sensor_msgs::msg::CameraInfo right_info_{};

  std::vector<std::uint8_t> left_gray_;
  std::vector<std::uint8_t> right_gray_;
  std::vector<std::uint8_t> left_rgb_;

  int preview_fps_ = 20;
  int jpeg_quality_ = 85;
  std::chrono::steady_clock::time_point last_preview_publish_{};
  bool has_preview_publish_ = false;

  std::atomic<bool> running_{false};
  std::thread capture_thread_;
  std::thread publication_thread_;
  std::thread auto_control_thread_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<StereoNode>();
    rclcpp::spin(node);
    node.reset();
  } catch (const std::exception& e) {
    RCLCPP_FATAL(
        rclcpp::get_logger("stereo_camera"), "%s", e.what());
    if (rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }

  if (rclcpp::ok()) rclcpp::shutdown();
  return 0;
}

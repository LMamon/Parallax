#include <parallax/camera/arducam_controls.hpp>
#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/frame_types.hpp>
#include <parallax/camera/stereo_camera.hpp>
#include <parallax/core/execution_context.hpp>
#include <parallax/isp/auto_control.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/isp/isp.hpp>
#include <parallax/isp/isp_config.hpp>
#include <parallax/stereo/calibration.hpp>
#include <parallax/stereo/rectification.hpp>

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
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

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

sensor_msgs::msg::CameraInfo scaledCameraInfo(
    const sensor_msgs::msg::CameraInfo& source,
    std::uint32_t width,
    std::uint32_t height) {
  auto info = source;
  const double sx =
      static_cast<double>(width) / static_cast<double>(source.width);
  const double sy =
      static_cast<double>(height) / static_cast<double>(source.height);

  info.width = width;
  info.height = height;

  info.k[0] *= sx;
  info.k[2] *= sx;
  info.k[4] *= sy;
  info.k[5] *= sy;

  info.p[0] *= sx;
  info.p[2] *= sx;
  info.p[3] *= sx;
  info.p[5] *= sy;
  info.p[6] *= sy;

  return info;
}

rclcpp::Time wallStamp(
    std::chrono::system_clock::time_point wall_timestamp) {
  return rclcpp::Time(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          wall_timestamp.time_since_epoch()).count(),
      RCL_SYSTEM_TIME);
}

struct RawSnapshot {
  std::vector<std::uint16_t> pixels;
  std::uint64_t sequence = 0;
  std::chrono::nanoseconds timestamp{0};
  std::chrono::system_clock::time_point wall_timestamp{};
  bool valid = false;
};

struct RectifiedHostSnapshot {
  std::vector<std::uint8_t> left_rgb;
  std::vector<std::uint8_t> right_rgb;
  std::uint64_t sequence = 0;
  std::chrono::system_clock::time_point wall_timestamp{};
};

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
    diagnostics_ = declare_parameter<bool>("diagnostics", false);

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

    auto isp_seed = isp_.acquireOutput();
    if (!isp_seed) {
      throw std::runtime_error("failed to acquire ISP initialization slot");
    }

    auto& preprocess = context_.preprocessLane();
    if (!rectifier_.initialize(
            calibration_,
            isp_seed->rgb,
            isp_seed->gray,
            preprocess.handle())) {
      throw std::runtime_error("failed to initialize stereo rectifier");
    }
    isp_seed.reset();

    initializeAutoControl();

    const auto qos = rclcpp::SensorDataQoS().keep_last(1);

    left_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "/stereo/left/image_rect", qos);
    right_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "/stereo/right/image_rect", qos);
    left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "/stereo/left/camera_info", qos);
    right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "/stereo/right/camera_info", qos);
    spatial_left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "/spatial/left/camera_info", qos);
    spatial_right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "/spatial/right/camera_info", qos);
    color_preview_pub_ =
        create_publisher<sensor_msgs::msg::CompressedImage>(
            "/stereo/left/image_rect_color/compressed", qos);

    left_info_ =
        makeRectifiedInfo(calibration_, calibration_.P1(), kLeftFrame);
    right_info_ =
        makeRectifiedInfo(calibration_, calibration_.P2(), kRightFrame);
    spatial_left_info_ = scaledCameraInfo(left_info_, 960U, 600U);
    spatial_right_info_ = scaledCameraInfo(right_info_, 960U, 600U);

    const auto raw_pixels =
        static_cast<std::size_t>(camera_config_.width) *
        static_cast<std::size_t>(camera_config_.height);
    for (auto& slot : raw_slots_) {
      slot.pixels.resize(raw_pixels);
    }

    const auto rect_pixels =
        static_cast<std::size_t>(calibration_.metadata().image_width) *
        static_cast<std::size_t>(calibration_.metadata().image_height);
    for (auto& slot : host_slots_) {
      slot.left_rgb.resize(rect_pixels * 3U);
      slot.right_rgb.resize(rect_pixels * 3U);
    }

    if (cudaStreamCreateWithFlags(
            &download_stream_, cudaStreamNonBlocking) != cudaSuccess) {
      throw std::runtime_error("failed to create ROS download stream");
    }

    running_.store(true);
    acquisition_thread_ =
        std::thread(&StereoNode::acquisitionLoop, this);
    compute_thread_ =
        std::thread(&StereoNode::computeLoop, this);
    mono_thread_ =
        std::thread(&StereoNode::monoPublicationLoop, this);
    preview_thread_ =
        std::thread(&StereoNode::previewPublicationLoop, this);
    auto_control_thread_ =
        std::thread(&StereoNode::autoControlLoop, this);

    RCLCPP_INFO(
        get_logger(),
        "AR0234 stereo ready: acquisition independent; ISP/rectification "
        "latest-value compute; mono ROS + %dHz JPEG preview",
        preview_fps_);
  }

  ~StereoNode() override {
    running_.store(false);
    raw_cv_.notify_all();
    host_cv_.notify_all();

    if (acquisition_thread_.joinable()) acquisition_thread_.join();
    if (compute_thread_.joinable()) compute_thread_.join();
    if (mono_thread_.joinable()) mono_thread_.join();
    if (preview_thread_.joinable()) preview_thread_.join();
    if (auto_control_thread_.joinable()) auto_control_thread_.join();

    (void)context_.drain();

    if (download_stream_ != nullptr) {
      cudaStreamSynchronize(download_stream_);
      cudaStreamDestroy(download_stream_);
      download_stream_ = nullptr;
    }

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
    auto diag_start = std::chrono::steady_clock::now();
    std::uint64_t diag_stats = 0;
    std::uint64_t diag_exposure = 0;
    std::uint64_t diag_gain = 0;
    std::uint64_t diag_wb = 0;

    while (running_.load() && rclcpp::ok() && auto_controller_) {
      parallax::isp::IspStatistics statistics{};

      if (isp_.tryGetStatistics(statistics)) {
        ++diag_stats;
        const auto update = auto_controller_->update(statistics);
        if (update.exposure_changed) ++diag_exposure;
        if (update.gain_changed) ++diag_gain;
        if (update.white_balance_changed) ++diag_wb;

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

      const auto diag_now = std::chrono::steady_clock::now();
      if (diagnostics_ &&
          diag_now - diag_start >= std::chrono::seconds(5)) {
        const double seconds =
            std::chrono::duration<double>(diag_now - diag_start).count();
        RCLCPP_INFO(
            get_logger(),
            "camera_diag auto_stats=%.2fHz exposure=%llu gain=%llu wb=%llu",
            static_cast<double>(diag_stats) / seconds,
            static_cast<unsigned long long>(diag_exposure),
            static_cast<unsigned long long>(diag_gain),
            static_cast<unsigned long long>(diag_wb));
        diag_stats = diag_exposure = diag_gain = diag_wb = 0;
        diag_start = diag_now;
      }

      std::this_thread::sleep_for(5ms);
    }
  }

  void acquisitionLoop() {
    unsigned failures = 0;
    std::uint64_t sequence = 0;

    auto boundary_window_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration boundary_capture_time{};
    std::chrono::steady_clock::duration boundary_post_capture_time{};
    std::uint64_t boundary_frames = 0;
    auto diag_start = std::chrono::steady_clock::now();
    std::uint64_t diag_frames = 0;
    std::chrono::nanoseconds diag_copy{0};

    while (running_.load() && rclcpp::ok()) {
      const auto compute_start = std::chrono::steady_clock::now();
      parallax::camera::RawFrame frame{};
      const auto boundary_capture_start = std::chrono::steady_clock::now();
      if (!camera_->capture(frame)) {
        if (++failures >= 10) {
          RCLCPP_ERROR(
              get_logger(), "camera repeatedly failed to capture");
          running_.store(false);
          raw_cv_.notify_all();
          host_cv_.notify_all();
          return;
        }
        continue;
      }

      const auto boundary_capture_end = std::chrono::steady_clock::now();
      failures = 0;

      const std::size_t expected_bytes =
          static_cast<std::size_t>(camera_config_.width) *
          static_cast<std::size_t>(camera_config_.height) *
          sizeof(std::uint16_t);

      if (frame.data == nullptr || frame.bytes < expected_bytes) {
        camera_->release(frame);
        RCLCPP_ERROR(
            get_logger(), "camera returned a short BA10 frame");
        continue;
      }

      int next = 0;
      {
        std::lock_guard<std::mutex> lock(raw_mutex_);
        next = 1 - raw_write_slot_;
      }

      auto& slot = raw_slots_[next];

      // Copy the MMAP frame, then immediately return the V4L2 buffer.
      // Acquisition never waits for ISP, VPI, ROS, JPEG, or Foxglove.
      const auto copy_start = std::chrono::steady_clock::now();
      std::memcpy(
          slot.pixels.data(), frame.data, expected_bytes);
      diag_copy += std::chrono::steady_clock::now() - copy_start;

      slot.timestamp = frame.timestamp;
      const auto steady_now = std::chrono::steady_clock::now();
      const auto system_now = std::chrono::system_clock::now();
      slot.wall_timestamp =
          system_now +
          (std::chrono::steady_clock::time_point{frame.timestamp} -
           steady_now);
      slot.sequence = ++sequence;
      slot.valid = true;

      if (!camera_->release(frame)) {
        RCLCPP_ERROR(get_logger(), "failed to requeue camera buffer");
        running_.store(false);
        raw_cv_.notify_all();
        host_cv_.notify_all();
        return;
      }

      const auto boundary_iteration_end = std::chrono::steady_clock::now();
      boundary_capture_time += boundary_capture_end - boundary_capture_start;
      boundary_post_capture_time += boundary_iteration_end - boundary_capture_end;
      ++boundary_frames;

      const auto boundary_window = boundary_iteration_end - boundary_window_start;
      if (diagnostics_ &&
          boundary_window >= std::chrono::seconds(5) &&
          boundary_frames != 0) {
        const double window_s =
            std::chrono::duration<double>(boundary_window).count();
        const double capture_ms =
            std::chrono::duration<double, std::milli>(
                boundary_capture_time).count() /
            static_cast<double>(boundary_frames);
        const double post_ms =
            std::chrono::duration<double, std::milli>(
                boundary_post_capture_time).count() /
            static_cast<double>(boundary_frames);

        RCLCPP_INFO(
            get_logger(),
            "camera_boundary dqbuf=%.3fms/frame post_capture=%.3fms/frame "
            "completed=%.2fHz frames=%llu",
            capture_ms,
            post_ms,
            static_cast<double>(boundary_frames) / window_s,
            static_cast<unsigned long long>(boundary_frames));

        boundary_window_start = boundary_iteration_end;
        boundary_capture_time = {};
        boundary_post_capture_time = {};
        boundary_frames = 0;
      }

      {
        std::lock_guard<std::mutex> lock(raw_mutex_);
        raw_write_slot_ = next;
        raw_sequence_ = slot.sequence;
      }
      raw_cv_.notify_one();
      ++diag_frames;

      const auto diag_now = std::chrono::steady_clock::now();
      if (diagnostics_ &&
          diag_now - diag_start >= std::chrono::seconds(5)) {
        const double seconds =
            std::chrono::duration<double>(diag_now - diag_start).count();
        const double copy_ms = diag_frames
            ? std::chrono::duration<double, std::milli>(diag_copy).count() /
                  static_cast<double>(diag_frames)
            : 0.0;
        RCLCPP_INFO(
            get_logger(), "camera_diag capture=%.2fHz raw_copy=%.3fms/frame",
            static_cast<double>(diag_frames) / seconds, copy_ms);
        diag_frames = 0;
        diag_copy = std::chrono::nanoseconds{0};
        diag_start = diag_now;
      }
    }
  }

  void computeLoop() {
    std::uint64_t last_raw = 0;
    auto diag_start = std::chrono::steady_clock::now();
    std::uint64_t diag_frames = 0;
    std::uint64_t diag_skipped = 0;
    std::chrono::nanoseconds diag_compute{0};

    while (running_.load() && rclcpp::ok()) {
      const auto compute_start = std::chrono::steady_clock::now();
      RawSnapshot raw;

      {
        std::unique_lock<std::mutex> lock(raw_mutex_);
        raw_cv_.wait(lock, [&] {
          return !running_.load() || raw_sequence_ > last_raw;
        });
        if (!running_.load()) return;

        const auto& latest = raw_slots_[raw_write_slot_];
        raw = latest;
        if (last_raw != 0 && latest.sequence > last_raw + 1) {
          diag_skipped += latest.sequence - last_raw - 1;
        }
        last_raw = latest.sequence;
      }

      parallax::camera::RawFrame frame{};
      frame.width = static_cast<std::uint32_t>(camera_config_.width);
      frame.height = static_cast<std::uint32_t>(camera_config_.height);
      frame.data = raw.pixels.data();
      frame.bytes = raw.pixels.size() * sizeof(std::uint16_t);
      frame.timestamp = raw.timestamp;

      auto isp_output = isp_.acquireOutput();
      if (!isp_output) continue;

      if (!isp_.process(frame, *isp_output) || !isp_.synchronize()) {
        RCLCPP_ERROR_THROTTLE(
            get_logger(), *get_clock(), 2000, "ISP processing failed");
        continue;
      }

      auto rectified = rectifier_.acquireOutput();
      if (!rectified) continue;

      auto& preprocess = context_.preprocessLane();
      if (!rectifier_.process(
              isp_output->rgb,
              isp_output->gray,
              *rectified,
              preprocess.handle()) ||
          !preprocess.synchronize()) {
        RCLCPP_ERROR_THROTTLE(
            get_logger(), *get_clock(), 2000, "rectification failed");
        continue;
      }

      int next = 0;
      {
        std::lock_guard<std::mutex> lock(host_mutex_);
        next = 1 - host_write_slot_;
      }
      auto& host = host_slots_[next];

      const auto rgb_pitch =
          static_cast<std::size_t>(rectified->rgb.width) * 3U;

      // Isaac ROS 3.2 DisparityNode accepts rgb8/bgr8, not mono8.
      // ROS publication and JPEG remain observation workers and cannot retain
      // accelerator/source ownership.
      if (!rectified->rgb.left.downloadAsync(
              host.left_rgb.data(), rgb_pitch, download_stream_) ||
          !rectified->rgb.right.downloadAsync(
              host.right_rgb.data(), rgb_pitch, download_stream_) ||
          cudaStreamSynchronize(download_stream_) != cudaSuccess) {
        RCLCPP_ERROR_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "rectified host staging failed");
        continue;
      }

      host.sequence = raw.sequence;
      host.wall_timestamp = raw.wall_timestamp;

      {
        std::lock_guard<std::mutex> lock(host_mutex_);
        host_write_slot_ = next;
        host_sequence_ = host.sequence;
      }
      host_cv_.notify_all();
      diag_compute += std::chrono::steady_clock::now() - compute_start;
      ++diag_frames;

      const auto diag_now = std::chrono::steady_clock::now();
      if (diagnostics_ &&
          diag_now - diag_start >= std::chrono::seconds(5)) {
        const double seconds =
            std::chrono::duration<double>(diag_now - diag_start).count();
        const double compute_ms = diag_frames
            ? std::chrono::duration<double, std::milli>(diag_compute).count() /
                  static_cast<double>(diag_frames)
            : 0.0;
        RCLCPP_INFO(
            get_logger(),
            "camera_diag compute=%.2fHz skipped=%llu total=%.3fms/frame",
            static_cast<double>(diag_frames) / seconds,
            static_cast<unsigned long long>(diag_skipped), compute_ms);
        diag_frames = 0;
        diag_skipped = 0;
        diag_compute = std::chrono::nanoseconds{0};
        diag_start = diag_now;
      }
    }
  }

  void monoPublicationLoop() {
    std::uint64_t last = 0;

    while (running_.load() && rclcpp::ok()) {
      RectifiedHostSnapshot frame;

      {
        std::unique_lock<std::mutex> lock(host_mutex_);
        host_cv_.wait(lock, [&] {
          return !running_.load() || host_sequence_ > last;
        });
        if (!running_.load()) return;

        frame = host_slots_[host_write_slot_];
        last = frame.sequence;
      }

      const auto width = calibration_.metadata().image_width;
      const auto height = calibration_.metadata().image_height;
      const auto stamp = wallStamp(frame.wall_timestamp);

      sensor_msgs::msg::Image left;
      left.header.stamp = stamp;
      left.header.frame_id = kLeftFrame;
      left.height = height;
      left.width = width;
      left.encoding = "rgb8";
      left.is_bigendian = false;
      left.step = width * 3U;
      left.data = std::move(frame.left_rgb);

      sensor_msgs::msg::Image right;
      right.header.stamp = stamp;
      right.header.frame_id = kRightFrame;
      right.height = height;
      right.width = width;
      right.encoding = "rgb8";
      right.is_bigendian = false;
      right.step = width * 3U;
      right.data = std::move(frame.right_rgb);

      left_info_.header.stamp = stamp;
      right_info_.header.stamp = stamp;
      spatial_left_info_.header.stamp = stamp;
      spatial_right_info_.header.stamp = stamp;

      left_image_pub_->publish(std::move(left));
      right_image_pub_->publish(std::move(right));
      left_info_pub_->publish(left_info_);
      right_info_pub_->publish(right_info_);
      spatial_left_info_pub_->publish(spatial_left_info_);
      spatial_right_info_pub_->publish(spatial_right_info_);
    }
  }

  void previewPublicationLoop() {
    using Clock = std::chrono::steady_clock;

    std::uint64_t last = 0;
    auto next_publish = Clock::now();

    while (running_.load() && rclcpp::ok()) {
      RectifiedHostSnapshot frame;

      {
        std::unique_lock<std::mutex> lock(host_mutex_);
        host_cv_.wait(lock, [&] {
          return !running_.load() || host_sequence_ > last;
        });
        if (!running_.load()) return;

        frame = host_slots_[host_write_slot_];
        last = frame.sequence;
      }

      const auto now = Clock::now();
      if (now < next_publish) continue;

      const auto width = calibration_.metadata().image_width;
      const auto height = calibration_.metadata().image_height;
      const std::size_t pitch =
          static_cast<std::size_t>(width) * 3U;

      cv::Mat rgb(
          static_cast<int>(height),
          static_cast<int>(width),
          CV_8UC3,
          frame.left_rgb.data(),
          pitch);
      cv::Mat bgr;
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

      std::vector<std::uint8_t> jpeg;
      const std::vector<int> parameters{
          cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};

      if (!cv::imencode(".jpg", bgr, jpeg, parameters)) continue;

      sensor_msgs::msg::CompressedImage message;
      message.header.stamp = wallStamp(frame.wall_timestamp);
      message.header.frame_id = kLeftFrame;
      message.format = "jpeg";
      message.data = std::move(jpeg);

      color_preview_pub_->publish(std::move(message));

      next_publish =
          now + std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(
                        1.0 / static_cast<double>(preview_fps_)));
    }
  }

  parallax::camera::CameraConfig camera_config_{};
  parallax::isp::IspConfig isp_config_{};
  parallax::stereo::StereoCalibration calibration_{};

  parallax::core::ExecutionContext context_{};
  std::unique_ptr<parallax::camera::StereoCamera> camera_;
  parallax::isp::ISP isp_{};
  parallax::stereo::StereoRectifier rectifier_{};
  std::unique_ptr<parallax::isp::AutoController> auto_controller_;

  std::array<RawSnapshot, 2> raw_slots_{};
  std::mutex raw_mutex_;
  std::condition_variable raw_cv_;
  int raw_write_slot_ = 0;
  std::uint64_t raw_sequence_ = 0;

  std::array<RectifiedHostSnapshot, 2> host_slots_{};
  std::mutex host_mutex_;
  std::condition_variable host_cv_;
  int host_write_slot_ = 0;
  std::uint64_t host_sequence_ = 0;

  cudaStream_t download_stream_ = nullptr;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr spatial_left_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr spatial_right_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr
      color_preview_pub_;

  sensor_msgs::msg::CameraInfo left_info_{};
  sensor_msgs::msg::CameraInfo right_info_{};
  sensor_msgs::msg::CameraInfo spatial_left_info_{};
  sensor_msgs::msg::CameraInfo spatial_right_info_{};

  int preview_fps_ = 20;
  int jpeg_quality_ = 85;
  bool diagnostics_ = false;

  std::atomic<bool> running_{false};
  std::thread acquisition_thread_;
  std::thread compute_thread_;
  std::thread mono_thread_;
  std::thread preview_thread_;
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

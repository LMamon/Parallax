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
#include <rclcpp_components/register_node_macro.hpp>
#include <isaac_ros_managed_nitros/managed_nitros_publisher.hpp>
#include <isaac_ros_nitros_image_type/nitros_image.hpp>
#include <isaac_ros_nitros/types/type_adapter_nitros_context.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <gxf/core/entity.hpp>
#include <gxf/core/gxf.h>
#include <gxf/multimedia/video.hpp>
#include <gxf/std/timestamp.hpp>
#pragma GCC diagnostic pop
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

namespace parallax::ros {

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

nvidia::isaac_ros::nitros::NitrosImage makePooledNitrosImage(
    const std::shared_ptr<parallax::stereo::StereoRectifier::OutputSlot>& owner,
    const parallax::cuda::CudaBuffer& buffer,
    const std_msgs::msg::Header& header) {
  using namespace nvidia;

  auto entity = gxf::Entity::New(
      isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext());
  if (!entity) {
    throw std::runtime_error("failed to create NITROS image entity");
  }

  auto video = entity->add<gxf::VideoBuffer>(header.frame_id.c_str());
  if (!video) {
    throw std::runtime_error("failed to add NITROS video buffer");
  }

  gxf::ColorPlane plane("RGB", 3, static_cast<std::uint32_t>(buffer.pitch()));
  plane.width = buffer.width();
  plane.height = buffer.height();
  plane.offset = 0;
  plane.size = buffer.allocatedBytes();

  gxf::VideoBufferInfo info{
      buffer.width(),
      buffer.height(),
      gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB,
      std::vector<gxf::ColorPlane>{plane},
      gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
  };

  auto held_owner = owner;
  auto wrapped = video.value()->wrapMemory(
      info,
      buffer.allocatedBytes(),
      gxf::MemoryStorageType::kDevice,
      const_cast<void*>(buffer.data()),
      [held_owner = std::move(held_owner)](void*) mutable {
        held_owner.reset();
        return gxf::Success;
      });
  if (!wrapped) {
    throw std::runtime_error("failed to wrap rectifier CUDA buffer for NITROS");
  }

  auto timestamp = entity->add<gxf::Timestamp>("timestamp");
  if (!timestamp) {
    throw std::runtime_error("failed to add NITROS image timestamp");
  }
  timestamp.value()->acqtime =
      static_cast<std::uint64_t>(header.stamp.sec) * 1000000000ULL +
      static_cast<std::uint64_t>(header.stamp.nanosec);

  isaac_ros::nitros::NitrosImage image{};
  image.handle = entity->eid();
  image.frame_id = header.frame_id;
  GxfEntityRefCountInc(
      isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext(),
      entity->eid());
  return image;
}

class StereoNode final : public rclcpp::Node {
 public:
  explicit StereoNode(const rclcpp::NodeOptions& options)
      : Node("stereo_camera", options) {
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

    left_nitros_pub_ = std::make_shared<
        nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
            nvidia::isaac_ros::nitros::NitrosImage>>(
        this,
        "/compute/stereo/left/image_rect",
        nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name,
        nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{},
        qos);
    right_nitros_pub_ = std::make_shared<
        nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
            nvidia::isaac_ros::nitros::NitrosImage>>(
        this,
        "/compute/stereo/right/image_rect",
        nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name,
        nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{},
        qos);

    left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "/stereo/left/camera_info", qos);
    right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "/stereo/right/camera_info", qos);
    spatial_left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "/spatial/left/camera_info", qos);
    spatial_right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "/spatial/right/camera_info", qos);

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

    running_.store(true);
    acquisition_thread_ =
        std::thread(&StereoNode::acquisitionLoop, this);
    compute_thread_ =
        std::thread(&StereoNode::computeLoop, this);
    auto_control_thread_ =
        std::thread(&StereoNode::autoControlLoop, this);

    RCLCPP_INFO(
        get_logger(),
        "AR0234 stereo ready: acquisition independent; ISP/rectification "
        "latest-value compute; direct pooled NITROS spatial ingress");
  }

  ~StereoNode() override {
    running_.store(false);
    raw_cv_.notify_all();

    if (acquisition_thread_.joinable()) acquisition_thread_.join();
    if (compute_thread_.joinable()) compute_thread_.join();
    if (auto_control_thread_.joinable()) auto_control_thread_.join();

    (void)context_.drain();

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

      std_msgs::msg::Header left_header;
      left_header.stamp = wallStamp(raw.wall_timestamp);
      left_header.frame_id = kLeftFrame;
      std_msgs::msg::Header right_header;
      right_header.stamp = left_header.stamp;
      right_header.frame_id = kRightFrame;

      try {
        auto left_image = makePooledNitrosImage(
            rectified, rectified->rgb.left, left_header);
        auto right_image = makePooledNitrosImage(
            rectified, rectified->rgb.right, right_header);

        left_nitros_pub_->publish(left_image);
        right_nitros_pub_->publish(right_image);
      } catch (const std::exception& e) {
        RCLCPP_ERROR_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "NITROS spatial publication failed: %s", e.what());
        continue;
      }

      // Resize synchronizes each image with its full-resolution CameraInfo.
      left_info_.header.stamp = left_header.stamp;
      right_info_.header.stamp = right_header.stamp;
      spatial_left_info_.header.stamp = left_header.stamp;
      spatial_right_info_.header.stamp = right_header.stamp;
      left_info_pub_->publish(left_info_);
      right_info_pub_->publish(right_info_);
      spatial_left_info_pub_->publish(spatial_left_info_);
      spatial_right_info_pub_->publish(spatial_right_info_);
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

  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> left_nitros_pub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> right_nitros_pub_;

  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr spatial_left_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr spatial_right_info_pub_;
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
  std::thread auto_control_thread_;
};

}  // namespace

RCLCPP_COMPONENTS_REGISTER_NODE(parallax::ros::StereoNode)

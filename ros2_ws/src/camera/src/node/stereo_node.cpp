#include <parallax/camera/stereo_node.hpp>

#include <rclcpp_components/register_node_macro.hpp>

#include <ctime>

namespace parallax::ros {

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

std::chrono::system_clock::time_point monotonicToSystemTime(
    std::chrono::nanoseconds monotonic_timestamp) {
  timespec monotonic_now_ts{};
  timespec realtime_now_ts{};

  if (::clock_gettime(CLOCK_MONOTONIC, &monotonic_now_ts) != 0 ||
      ::clock_gettime(CLOCK_REALTIME, &realtime_now_ts) != 0) {
    throw std::runtime_error("clock_gettime failed");
  }

  const auto monotonic_now =
      std::chrono::seconds(monotonic_now_ts.tv_sec) +
      std::chrono::nanoseconds(monotonic_now_ts.tv_nsec);

  const auto realtime_now =
      std::chrono::seconds(realtime_now_ts.tv_sec) +
      std::chrono::nanoseconds(realtime_now_ts.tv_nsec);

  const auto realtime_timestamp =
      realtime_now + (monotonic_timestamp - monotonic_now);

  return std::chrono::system_clock::time_point{
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          realtime_timestamp)};
}

StereoNode::StereoNode(const rclcpp::NodeOptions& options)
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

  if (!spatial_gray_pool_.initialize([](SpatialGraySlot& slot, std::size_t) {
            slot.frame.width = 960U;
            slot.frame.height = 600U;

            return slot.frame.left.allocate(960U, 600U, 1U, sizeof(std::uint8_t)) &&
                slot.frame.right.allocate(960U, 600U, 1U, sizeof(std::uint8_t));
          })) {
    throw std::runtime_error("failed to initialize spatial gray pool");
  }

  const auto* spatial_prototype = spatial_gray_pool_.prototype();

  if (spatial_prototype == nullptr || !matcher_.initialize(spatial_prototype->frame,
          context_.stereoLane().handle())) {

    throw std::runtime_error("failed to initialize CUDA stereo matcher");
  }

  if (!depth_pool_.initialize([](parallax::isp::DepthFrame& depth, std::size_t index) {
        depth.width = 960U;
        depth.height = 600U;
        depth.storage_slot = static_cast<std::uint32_t>(index);
        return depth.depth.allocate(960U, 600U, 1U, sizeof(float));
      })) {
    throw std::runtime_error("failed to allocate spatial depth pool");
  }

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
  depth_nitros_pub_ = std::make_shared<
      nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
          nvidia::isaac_ros::nitros::NitrosImage>>(
      this,
      "/stereo/depth",
      nvidia::isaac_ros::nitros::nitros_image_32FC1_t::supported_type_name,
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

  if (cudaEventCreateWithFlags(
          &raw_consumed_event_, cudaEventDisableTiming) != cudaSuccess) {
    throw std::runtime_error("failed to create raw capture completion event");
  }

  running_.store(true);
  acquisition_thread_ =
      std::thread(&StereoNode::acquisitionLoop, this);
  compute_thread_ =
      std::thread(&StereoNode::computeLoop, this);
  stereo_thread_ =
      std::thread(&StereoNode::stereoLoop, this);
  rgb_thread_ =
    std::thread(&StereoNode::rgbPublishLoop, this);
  depth_thread_ =
    std::thread(&StereoNode::depthPublishLoop, this);
  auto_control_thread_ =
      std::thread(&StereoNode::autoControlLoop, this);

  RCLCPP_INFO(
      get_logger(),
      "AR0234 stereo ready: acquisition independent; ISP/rectification "
      "latest-value compute; direct pooled NITROS spatial ingress");
}

StereoNode::~StereoNode() {
  running_.store(false);
  raw_cv_.notify_all();
  stereo_cv_.notify_all();
  rgb_cv_.notify_all();
  depth_cv_.notify_all();

  if (acquisition_thread_.joinable()) acquisition_thread_.join();
  if (compute_thread_.joinable()) compute_thread_.join();
  if (stereo_thread_.joinable()) stereo_thread_.join();
  if (auto_control_thread_.joinable()) auto_control_thread_.join();
  if (rgb_thread_.joinable()) rgb_thread_.join();
  if (depth_thread_.joinable()) depth_thread_.join();

  {
    std::lock_guard<std::mutex> lock(raw_mutex_);
    if (raw_pending_.valid && camera_) {
      (void)camera_->release(raw_pending_.frame);
      raw_pending_.valid = false;
    }
  }

  if (raw_consumed_event_ != nullptr) {
    cudaEventDestroy(raw_consumed_event_);
    raw_consumed_event_ = nullptr;
  }

  (void)context_.drain();

  {
    std::lock_guard<std::mutex> lock(stereo_mutex_);
    stereo_pending_.reset();
  }

  auto_controller_.reset();
  matcher_.shutdown();
  depth_pool_.reset();
  spatial_gray_pool_.reset();
  rectifier_.shutdown();
  isp_.shutdown();

  if (camera_) {
    camera_->shutdown();
    camera_.reset();
  }

  context_.shutdown();
}

}  // namespace parallax::ros

RCLCPP_COMPONENTS_REGISTER_NODE(parallax::ros::StereoNode)

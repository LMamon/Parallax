#pragma once

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

#include <rclcpp/rclcpp.hpp>
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

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace parallax::ros {

inline constexpr char kLeftFrame[] = "left_camera_optical_frame";
inline constexpr char kRightFrame[] = "right_camera_optical_frame";

sensor_msgs::msg::CameraInfo makeRectifiedInfo(const parallax::stereo::StereoCalibration& calibration,
                                               const std::array<double, 12>& projection,
                                               const char* frame_id);

sensor_msgs::msg::CameraInfo scaledCameraInfo(const sensor_msgs::msg::CameraInfo& source,
                                              std::uint32_t width,
                                              std::uint32_t height);

rclcpp::Time wallStamp(std::chrono::system_clock::time_point wall_timestamp);

std::chrono::system_clock::time_point monotonicToSystemTime(std::chrono::nanoseconds monotonic_timestamp);

struct RawLease {
  parallax::camera::RawFrame frame{};
  std::uint64_t sequence = 0;
  std::chrono::system_clock::time_point wall_timestamp{};
  bool valid = false;
};

struct RectifiedHostSnapshot {
  std::vector<std::uint8_t> left_rgb;
  std::vector<std::uint8_t> right_rgb;
  std::uint64_t sequence = 0;
  std::chrono::system_clock::time_point wall_timestamp{};
};

template <typename Owner>
nvidia::isaac_ros::nitros::NitrosImage makePooledNitrosImage(const std::shared_ptr<Owner>& owner,
                                                             const parallax::cuda::CudaBuffer& buffer,
                                                             const std_msgs::msg::Header& header,
                                                             nvidia::gxf::VideoFormat format,
                                                             const char* plane_name,
                                                             std::uint8_t bytes_per_pixel) {
  
    using namespace nvidia;

    auto entity = gxf::Entity::New(isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext());
    if (!entity) throw std::runtime_error("failed to create NITROS image entity");

    auto video = entity->add<gxf::VideoBuffer>(header.frame_id.c_str());
    if (!video) throw std::runtime_error("failed to add NITROS video buffer");

    gxf::ColorPlane plane(plane_name, bytes_per_pixel, static_cast<std::uint32_t>(buffer.pitch()));
    plane.width = buffer.width();
    plane.height = buffer.height();
    plane.offset = 0;
    plane.size = buffer.allocatedBytes();

    gxf::VideoBufferInfo info{buffer.width(),
                              buffer.height(),
                              format,
                              std::vector<gxf::ColorPlane>{plane},
                              gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};

    auto held_owner = owner;
    auto wrapped = video.value()->wrapMemory(info,
                                            buffer.allocatedBytes(),
                                            gxf::MemoryStorageType::kDevice,
                                            const_cast<void*>(buffer.data()),
                                            [held_owner = std::move(held_owner)](void*) mutable {
                                                held_owner.reset();
                                                return gxf::Success;
                                            });

    if (!wrapped) throw std::runtime_error("failed to wrap CUDA buffer for NITROS");

    auto timestamp = entity->add<gxf::Timestamp>("timestamp");
    if (!timestamp) throw std::runtime_error("failed to add NITROS timestamp");
    timestamp.value()->acqtime = static_cast<std::uint64_t>(header.stamp.sec) * 1000000000ULL +
                                 static_cast<std::uint64_t>(header.stamp.nanosec);

    isaac_ros::nitros::NitrosImage image{};
    image.handle = entity->eid();
    image.frame_id = header.frame_id;
    GxfEntityRefCountInc(isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext(),
                         entity->eid());
    return image;
}

    class StereoNode final : public rclcpp::Node {
        public:
            explicit StereoNode(const rclcpp::NodeOptions& options);
            ~StereoNode() override;

        private:
            using IspOutput = parallax::isp::ISP::OutputSlot;
            using RectifiedOutput = parallax::stereo::StereoRectifier::OutputSlot;
            using Completion = parallax::core::CompletionHandle;

            struct RgbPublication {
                std::shared_ptr<IspOutput> isp;
                std::shared_ptr<RectifiedOutput> frame;
                Completion ready{};
                std_msgs::msg::Header left{};
                std_msgs::msg::Header right{};
            };

            struct PreviewPublication {
                std::shared_ptr<RectifiedOutput> frame;
                std_msgs::msg::Header left{};
                std_msgs::msg::Header right{};
                bool valid = false;
            };

            static constexpr std::size_t RgbQueueCapacity = 4;

            std::array<RgbPublication, RgbQueueCapacity> rgb_queue_{};
            std::size_t rgb_head_ = 0;
            std::size_t rgb_tail_ = 0;
            std::size_t rgb_size_ = 0;
            std::mutex rgb_mutex_;
            std::condition_variable rgb_cv_;
            std::thread rgb_thread_;

            PreviewPublication preview_pending_{};
            std::mutex preview_mutex_;
            std::condition_variable preview_cv_;
            std::thread preview_thread_;
            cudaStream_t preview_stream_ = nullptr;
            std::chrono::steady_clock::time_point preview_next_{};

            void rgbPublishLoop();
            void previewLoop();
            void initializeAutoControl();
            void autoControlLoop();
            void acquisitionLoop();
            void computeLoop();

            parallax::camera::CameraConfig camera_config_{};
            parallax::isp::IspConfig isp_config_{};
            parallax::stereo::StereoCalibration calibration_{};

            parallax::core::ExecutionContext context_{};
            std::unique_ptr<parallax::camera::StereoCamera> camera_;
            parallax::isp::ISP isp_{};
            parallax::stereo::StereoRectifier rectifier_{};
            std::unique_ptr<parallax::isp::AutoController> auto_controller_;

            RawLease raw_pending_{};
            std::mutex raw_mutex_;
            std::condition_variable raw_cv_;
            cudaEvent_t raw_consumed_event_ = nullptr;

            std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                nvidia::isaac_ros::nitros::NitrosImage>> left_nitros_pub_;
                
            std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                nvidia::isaac_ros::nitros::NitrosImage>> right_nitros_pub_;

            rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr preview_left_pub_;
            rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr preview_right_pub_;
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

}
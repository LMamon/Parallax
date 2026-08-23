#include <parallax/visualization/publisher.hpp>
#include <parallax/pose/charuco_pose.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <cstring>
#include <iostream>
#include <chrono>
#include <array>
#include <iomanip>
#include <cmath>

namespace parallax::visualization {

    namespace {
        constexpr const char* kFrameId = "camera_left_optical";

        bool checkFoxglove(const foxglove::FoxgloveError& error, const char* message) {
            if (error != foxglove::FoxgloveError::Ok) {
                std::cerr << message << ": "
                        << foxglove::strerror(error) << '\n';

                return false;
            }
            return true;
        }

        foxglove::messages::Timestamp nowTimestamp() {
            const auto now = std::chrono::system_clock::now();
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                now.time_since_epoch()).count();

            foxglove::messages::Timestamp timestamp;
            timestamp.sec = static_cast<std::int32_t>(ns / 1'000'000'000LL);
            timestamp.nsec = static_cast<std::uint32_t>(ns % 1'000'000'000LL);

            return timestamp;
        }
    }

    Publisher::~Publisher() { shutdown(); }

    bool Publisher::initialize(std::uint32_t width, std::uint32_t height, std::uint32_t fps) {
        if (initialized_) return true;

        if (width == 0 || height == 0 || fps == 0) {
            std::cerr << "Invalid visualization dimensions/FPS\n";
            return false;
        }

        width_ = width;
        height_ = height;
        fps_ = fps;

        auto left_result = foxglove::messages::CompressedVideoChannel::create("/camera/left/image");
        if (!left_result.has_value()) {
            std::cerr
                << "Failed to create /camera/left/image channel: "
                << foxglove::strerror(left_result.error()) << '\n';

            shutdown();
            return false;
        }
        left_image_channel_.emplace(std::move(left_result.value()));

        auto calibration_result = foxglove::messages::CameraCalibrationChannel::create("/camera/left/calibration");
        if (!calibration_result.has_value()) {
            std::cerr
                << "Failed to create /camera/left/calibration channel: "
                << foxglove::strerror(calibration_result.error()) << '\n';

            shutdown();
            return false;
        }
        left_calibration_channel_.emplace(std::move(calibration_result.value()));

        auto disparity_result = foxglove::messages::RawImageChannel::create("/stereo/disparity");
        if (!disparity_result.has_value()) {
            std::cerr
                << "Failed to create /stereo/disparity channel: "
                << foxglove::strerror(disparity_result.error()) << '\n';

            shutdown();
            return false;
        }
        disparity_channel_.emplace(std::move(disparity_result.value()));

        auto confidence_result = foxglove::messages::RawImageChannel::create("/stereo/confidence");
        if (!confidence_result.has_value()) {
            std::cerr << "Failed to create /stereo/confidence channel: "
                    << foxglove::strerror(confidence_result.error()) << '\n';

            shutdown();
            return false;
        }
        confidence_channel_.emplace(std::move(confidence_result.value()));

        auto depth_result = foxglove::messages::RawImageChannel::create("/stereo/depth");
        if (!depth_result.has_value()) {
            std::cerr << "Failed to create /stereo/depth: "
                    << foxglove::strerror(depth_result.error()) << '\n';
            
            shutdown();
            return false;
        }
        depth_channel_.emplace(std::move(depth_result.value()));

        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            std::cerr << "Failed to create visualization CUDA stream\n";
            shutdown();
            return false;
        }

        const std::size_t pixels = static_cast<std::size_t>(width_) * height_;
        const std::size_t rgb_bytes = pixels * 3 * sizeof(std::uint8_t);

        const std::size_t disparity_bytes = pixels * sizeof(std::int16_t);
        const std::size_t confidence_bytes = pixels * sizeof(std::uint16_t);
        const std::size_t depth_bytes = pixels * sizeof(float);


        if (cudaMallocHost(reinterpret_cast<void**>(&host_rgb_), rgb_bytes) != cudaSuccess) {
            std::cerr << "Failed to allocate RGB staging buffer\n";
            shutdown();
            return false;
        }

        if (cudaMallocHost(reinterpret_cast<void**>(&host_disparity_), disparity_bytes) != cudaSuccess) {
            std::cerr << "Failed to allocate disparity staging buffer\n";
            shutdown();
            return false;
        }

        if (cudaMallocHost(reinterpret_cast<void**>(&host_confidence_), confidence_bytes) != cudaSuccess) {
            std::cerr << "Failed to allocate confidence staging buffer\n";
            shutdown();
            return false;
        }

        if (cudaMallocHost(reinterpret_cast<void**>(&host_depth_), depth_bytes) != cudaSuccess) {
            std::cerr << "Failed to allocate depth staging buffer\n";
            shutdown();
            return false;
        }

        disparity_float_.resize(pixels);

        if (!video_encoder_.initialize(width_, height_, fps_)) {
            std::cerr << "Failed to initialize video encoder\n";
            shutdown();
            return false;
        }

        initialized_ = true;
        return true;
    }

    bool Publisher::publishLeftCalibration(const parallax::stereo::StereoCalibration& calibration) {
        if (!initialized_ ||
            !left_calibration_channel_ ||
            !calibration.loaded()) {
            return false;
        }

        const auto& metadata = calibration.metadata();
        const auto& p1 = calibration.P1();

        foxglove::messages::CameraCalibration message;

        message.frame_id = kFrameId;

        message.width = metadata.image_width;
        message.height = metadata.image_height;

        // The published image is already rectified.
        message.distortion_model = "plumb_bob";
        message.d = {0.0, 0.0, 0.0, 0.0, 0.0};

        // Intrinsics of the rectified virtual camera.
        message.k = {p1[0], p1[1], p1[2],
                    p1[4], p1[5], p1[6],
                    p1[8], p1[9], p1[10]};

        // Image has already undergone the stereo rectification transform.
        message.r = {1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0};

        message.p = p1;

        return checkFoxglove(left_calibration_channel_->log(message), "Failed to publish /camera/left/calibration");
    }

    bool Publisher::publishLeftImage(const parallax::isp::RectifiedStereoFrame& frame,
                                     const parallax::pose::CharucoPoseResult& pose,
                                     std::chrono::steady_clock::time_point timestamp) {

        if (!initialized_ || !left_image_channel_ || !frame.left.isAllocated()) {
            return false;
        }

        if (frame.width != width_ || frame.height != height_) {
            std::cerr << "Visualization RGB dimensions changed\n";
            return false;
        }

        const std::size_t host_pitch = static_cast<std::size_t>(width_) *
                                       parallax::isp::RectifiedStereoFrame::Channels *
                                       sizeof(std::uint8_t);

        if (!frame.left.downloadAsync(host_rgb_, host_pitch, stream_)) {
            std::cerr << "Failed to download left RGB frame\n";
            return false;
        }

        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            std::cerr << "Failed to synchronize RGB download\n";
            return false;
        }

        cv::Mat image(static_cast<int>(height_),
                      static_cast<int>(width_),
                      CV_8UC3,
                      host_rgb_,
                      host_pitch);

        if (pose.pose_valid) {
            std::vector<cv::Point> polygon;
            polygon.reserve(4);

            for (const auto& p : pose.projected_plane) {
                polygon.emplace_back(static_cast<int>(std::lround(p.x)),
                                     static_cast<int>(std::lround(p.y)));
            }
            cv::polylines(image, polygon, true, cv::Scalar(0, 255, 0), 5, cv::LINE_AA);

            cv:circle(image, 
                      cv::Point(static_cast<int>(std::lround(pose.projected_center.x)),
                            static_cast<int>(std::lround(pose.projected_center.y))),
                      8,
                      cv::Scalar(255, 0, 0),
                      -1);
        }
    

        const std::size_t rgb_bytes = host_pitch * height_;

        if (!video_encoder_.encode(host_rgb_, rgb_bytes, encoded_video_)) {
            return false;
        }

        foxglove::messages::CompressedVideo message;

        message.timestamp = nowTimestamp();
        message.frame_id = kFrameId;
        message.format = "h264";
        message.data = encoded_video_;

        return checkFoxglove(left_image_channel_->log(message), "Failed to publish /camera/left/image");
    }

    bool Publisher::publishDepth(const parallax::isp::DepthFrame& frame) {
        if (!initialized_ || !depth_channel_ || !frame.depth.isAllocated()) {
            return false;
        }

        if (frame.width != width_ || frame.height != height_) {
            std::cerr << "Visualization depth dimensions changed\n";
            return false;
        }

        const std::size_t host_pitch = static_cast<std::size_t>(width_) * sizeof(float);

        if (!frame.depth.downloadAsync(host_depth_, host_pitch, stream_)) {
            std::cerr << "Failed to download left RGB frame\n";
            return false;
        }

        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            std::cerr << "Failed to synchronize RGB download\n";
            return false;
        }

        const std::size_t bytes = static_cast<std::size_t>(frame.width) * frame.height * sizeof(float);

        foxglove::messages::RawImage message;

        message.timestamp = nowTimestamp();
        message.frame_id = kFrameId;
        message.width = frame.width;
        message.height = frame.height;
        message.encoding = "32FC1";
        message.step = frame.width * sizeof(float);

        message.data.resize(bytes);

        std::memcpy(message.data.data(), host_depth_, bytes);

        return checkFoxglove(depth_channel_->log(message), "Failed to publish /stereo/depth");
    }

    bool Publisher::publishDisparity(const parallax::isp::StereoMatchFrame& frame) {
        if (!initialized_ || !disparity_channel_ || !frame.disparity.isAllocated()) {
            return false;
        }

        const std::size_t host_pitch = static_cast<std::size_t>(frame.width) * sizeof(std::int16_t);

        if (!frame.disparity.downloadAsync(host_disparity_, host_pitch, stream_)) {
            std::cerr << "Failed to download disparity\n";
            return false;
        }

        if (cudaStreamSynchronize(stream_) != cudaSuccess) { return false; }

        const std::size_t pixels = static_cast<std::size_t>(frame.width) * frame.height;

        for (std::size_t i = 0; i < pixels; ++i) {
            disparity_float_[i] = static_cast<float>(host_disparity_[i]) /
                                    parallax::isp::StereoMatchFrame::DisparityScale;
        }

        foxglove::messages::RawImage message;

        message.frame_id = kFrameId;
        message.width = frame.width;
        message.height = frame.height;
        message.encoding = "32FC1";
        message.step = frame.width * sizeof(float);

        message.data.resize(pixels * sizeof(float));

        std::memcpy( message.data.data(), disparity_float_.data(), message.data.size());

        return checkFoxglove(disparity_channel_->log(message), "Failed to publish /stereo/disparity");
    }

    bool Publisher::publishConfidence(const parallax::isp::StereoMatchFrame& frame) {
        if (!initialized_ || !confidence_channel_ || !frame.confidence.isAllocated()) {
            return false;
        }

        const std::size_t host_pitch = static_cast<std::size_t>(frame.width) * sizeof(std::uint16_t);

        if (!frame.confidence.downloadAsync(host_confidence_, host_pitch, stream_)) {
            std::cerr << "Failed to download confidence\n";
            return false;
        }

        if (cudaStreamSynchronize(stream_) != cudaSuccess) { return false; }

        const std::size_t bytes = static_cast<std::size_t>(frame.width) *
                                    frame.height *
                                    sizeof(std::uint16_t);

        foxglove::messages::RawImage message;

        message.frame_id = kFrameId;
        message.width = frame.width;
        message.height = frame.height;
        message.encoding = "16UC1";
        message.step = frame.width * sizeof(std::uint16_t);

        message.data.resize(bytes);

        std::memcpy(message.data.data(), host_confidence_, bytes);

        return checkFoxglove(confidence_channel_->log(message), "Failed to publish /stereo/confidence");
    }

    void Publisher::shutdown() {
        video_encoder_.shutdown();

        if (left_image_channel_) {
            left_image_channel_->close();
            left_image_channel_.reset();
        }
        if (left_calibration_channel_) {
            left_calibration_channel_->close();
            left_calibration_channel_.reset();
        }

        if (depth_channel_) {
            depth_channel_->close();
            depth_channel_.reset();
        }

        if (disparity_channel_) {
            disparity_channel_->close();
            disparity_channel_.reset();
        }

        if (confidence_channel_) {
            confidence_channel_->close();
            confidence_channel_.reset();
        }

        if (host_rgb_ != nullptr) {
            cudaFreeHost(host_rgb_);
            host_rgb_ = nullptr;
        }

        if (host_depth_ != nullptr) {
            cudaFreeHost(host_depth_);
            host_depth_ = nullptr;
        }

        if (host_disparity_ != nullptr) {
            cudaFreeHost(host_disparity_);
            host_disparity_ = nullptr;
        }

        if (host_confidence_ != nullptr) {
            cudaFreeHost(host_confidence_);
            host_confidence_ = nullptr;
        }

        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }

        disparity_float_.clear();
        encoded_video_.clear();

        width_ = 0;
        height_ = 0;
        fps_ = 0;

        initialized_ = false;
    }
}
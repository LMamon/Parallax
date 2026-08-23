#include <parallax/pose/charuco_pose.hpp>

#include <iostream>

namespace parallax::pose {
    CharucoPose::CharucoPose() : dictionary_(cv::aruco::getPredefinedDictionary(
                                            cv::aruco::DICT_4X4_50)),
                                            board_(cv::Size(11, 8),
                                            0.024f,
                                            0.018f,
                                            dictionary_),
                                            detector_(board_) {}
    
    CharucoPose::~CharucoPose() { shutdown(); }

    bool CharucoPose::initialize(std::uint32_t width, 
                                std::uint32_t height,
                                const parallax::stereo::StereoCalibration& calibration) {

        if (initialized_) return true;
        if (width == 0 || height == 0) {
            std::cerr << "CharucoPose: invalid image dimensions\n";
            return false;
        }

        width_ = width;
        height_ = height;
        const auto&p = calibration.P1();

        camera_matrix_ = cv::Matx33d(p[0], p[1], p[2],
                                     p[4], p[5], p[6],
                                     p[8], p[9], p[10]);

        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            std::cerr << "CharucoPose: failed to create CUDA stream\n";
            shutdown();
            return false;
        }

        const std::size_t bytes = static_cast<std::size_t>(width_) * height_;
        if (cudaMallocHost(reinterpret_cast<void**>(&host_gray_), bytes) != cudaSuccess) {
            std::cerr << "CharucoPose: failed to allocate grayscale staging buffer\n";
            shutdown();
            return false;
        }
        
        initialized_ = true;
        return true;
    }

    bool CharucoPose::process(const parallax::isp::RectifiedStereoGrayFrame& frame,
                            CharucoPoseResult& result) {
        
        if (!initialized_) return false;
        result.board_detected = false;
        result.charuco_corners.clear();
        result.charuco_ids.clear();

        if (!frame.left.isAllocated()) { return false; }
        if (frame.width != width_ || frame.height != height_) { return false; }

        const std::size_t host_pitch = static_cast<std::size_t>(width_) * sizeof(std::uint8_t);

        if (!frame.left.downloadAsync(host_gray_, host_pitch, stream_)) {
            std::cerr << "CharucoPose: failed to download grayscale frame\n";
            return false;
        }

        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            std::cerr << "CharucoPose: grayscale download synchronization failed\n";
            return false;
        }

        cv::Mat gray(static_cast<int>(height_), static_cast<int>(width_), CV_8UC1, host_gray_);
        detector_.detectBoard(gray, result.charuco_corners, result.charuco_ids);

        result.board_detected = result.charuco_ids.size() >= 6;
        result.pose_valid = false;
        result.plane_valid = false;
        result.depth_valid = false;
        result.depth_m = 0.0f;
        
        if (!result.board_detected) return true;

        const auto& board_corners = board_.getChessboardCorners();

        std::vector<cv::Point3f> object_points;
        object_points.reserve(result.charuco_ids.size());

        for (const int id : result.charuco_ids) {
            if (id < 0 || static_cast<std::size_t>(id) >= board_corners.size()) {
                return false;
            }

            object_points.push_back(board_corners[id]);
        }

        try {
            result.pose_valid = cv::solvePnP(object_points,
                                            result.charuco_corners,
                                            camera_matrix_,
                                            distortion_,
                                            result.rvec,
                                            result.tvec,
                                            false,
                                            cv::SOLVEPNP_ITERATIVE);
        } catch (const cv::Exception& e) {
            std::cerr << "CharucoPose: solvePnP failed: " << e.what() << "\n";
            result.pose_valid = false;
            return true;
        }
        if (!result.pose_valid) return true;

        constexpr float kSquareLength = 0.024f;
        constexpr float kBoardWidth = 11.0f * kSquareLength;
        constexpr float kBoardHeight= 8.0f * kSquareLength;

        const std::vector<cv::Point3f> plane_points{{0.0f, 0.0f, 0.0f},
                                                    {kBoardWidth, 0.0f, 0.0f},
                                                    {kBoardWidth, kBoardHeight, 0.0f},
                                                    {0.0f, kBoardHeight, 0.0f}};
                                                    

        const std::vector<cv::Point3f> center_point{{kBoardWidth * 0.5f, kBoardHeight * 0.5f, 0.0f}};
        std::vector<cv::Point2f> projected_plane;
        std::vector<cv::Point2f> projected_center;

        cv::projectPoints(plane_points,
                          result.rvec,
                          result.tvec,
                          camera_matrix_,
                          distortion_,
                          projected_plane);

        cv::projectPoints(center_point,
                          result.rvec,
                          result.tvec,
                          camera_matrix_,
                          distortion_,
                          projected_center);


        if (projected_plane.size() == 4 && projected_center.size() == 1) {
            for (std::size_t i = 0; i < 4; ++i) {
                result.projected_plane[i] = projected_plane[i];
            }
            result.projected_center = projected_center[0];
            result.plane_valid = true;
        }

        return true;
    }

    void CharucoPose::shutdown() {
        if (host_gray_ != nullptr) {
            cudaFreeHost(host_gray_);
            host_gray_ = nullptr;
        }

        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }

        width_ = 0;
        height_ = 0;
        initialized_ = false;
    }
}
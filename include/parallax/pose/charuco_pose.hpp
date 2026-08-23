#pragma once

#include <parallax/isp/frame_types.hpp>
#include <parallax/stereo/calibration.hpp>

#include <opencv2/aruco.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>
#include <array>


namespace parallax::pose {
    struct CharucoPoseResult {
        bool board_detected = false;
        bool pose_valid = false;
        bool plane_valid = false;

        bool depth_valid = false;
        float depth_m = 0.0f;
        
        std::vector<cv::Point2f> charuco_corners;
        std::vector<int> charuco_ids;

        cv::Vec3d rvec{};
        cv::Vec3d tvec{};

        std::array<cv::Point2f, 4> projected_plane{};
        cv::Point2f projected_center{};
    };

    class CharucoPose {
        public:
            CharucoPose();
            ~CharucoPose();

            CharucoPose(const CharucoPose&) = delete;
            CharucoPose& operator=(const CharucoPose&) = delete;

            bool initialize(std::uint32_t width, 
                            std::uint32_t height, 
                            const parallax::stereo::StereoCalibration& calibration);

            bool process(const parallax::isp::RectifiedStereoGrayFrame& frame, 
                        CharucoPoseResult& result);



            void shutdown();

            [[nodiscard]] bool initialized() const noexcept { return initialized_; }

        private:
            cv::aruco::Dictionary dictionary_;
            cv::aruco::CharucoBoard board_;
            cv::aruco::CharucoDetector detector_;
            cv::Matx33d camera_matrix_{};
            // already consuming rectified imaged
            cv::Vec<double, 5> distortion_{0.0, 0.0, 0.0, 0.0, 0.0,};

            cudaStream_t stream_ = nullptr;
            std::uint8_t* host_gray_ = nullptr;
            
            std::uint32_t width_ = 0;
            std::uint32_t height_ = 0;
                
            bool initialized_ = false;
    };
}
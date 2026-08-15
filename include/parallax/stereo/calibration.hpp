#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace parallax::stereo {

    struct StereoCalibrationMetadata {
        std::uint32_t image_width = 0;
        std::uint32_t image_height = 0;

        double virtual_fx = 0.0;
        double virtual_fy = 0.0;
        double virtual_cx = 0.0;
        double virtual_cy = 0.0;
        double rectification_alpha = 0.0;
        double baseline_mm = 0.0;

        std::string extrinsics_convention;
    };

    class StereoCalibration {
        public:
            bool load(const std::filesystem::path& directory);
            //Accessors to calibration/rectification data will go here
            const StereoCalibrationMetadata& metadata() const { return metadata_; }

            const std::array<double, 12>& P1() const { return p1_; }
            const std::array<double, 12>& P2() const { return p2_; }
            const std::array<double, 16>& Q()  const { return q_; }
            const std::array<double, 9>& R1()  const { return r1_; }
            const std::array<double, 9>& R2()  const { return r2_; }

            const std::vector<float>& leftMapX() const { return left_map_x_; }
            const std::vector<float>& leftMapY() const { return left_map_y_; }
            const std::vector<float>& rightMapX() const { return right_map_x_; }
            const std::vector<float>& rightMapY() const { return right_map_y_; }

            bool loaded() const { return loaded_; }

        private:
            StereoCalibrationMetadata metadata_;

            std::array<double, 12> p1_{};
            std::array<double, 12> p2_{};
            std::array<double, 16> q_{};
            std::array<double, 9> r1_{};
            std::array<double, 9> r2_{};

            std::vector<float> left_map_x_;
            std::vector<float> left_map_y_;
            std::vector<float> right_map_x_;
            std::vector<float> right_map_y_;

            bool loaded_ = false;
    };
}
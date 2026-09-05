#pragma once

#include <parallax/perception/image_space.hpp>
#include <parallax/stereo/calibration.hpp>

#include <opencv2/core/types.hpp>

#include <cstdint>
#include <vector>

namespace parallax::perception {

    class ImageSpaceMapper {
        public:
            bool initialize(const stereo::StereoCalibration& calibration);
            [[nodiscard]] bool initialized() const noexcept { return initialized_; }

            bool initialize(std::uint32_t width,
                            std::uint32_t height,
                            const std::vector<float>& rectified_to_rgb_x,
                            const std::vector<float>& rectified_to_rgb_y);

            [[nodiscard]] bool mapPoint(const cv::Point2f& point,
                                        ImageSpace source,
                                        ImageSpace destination,
                                        cv::Point2f& mapped) const noexcept;

            [[nodiscard]] bool mapRect(const cv::Rect2f& rect,
                                       ImageSpace source,
                                       ImageSpace destination,
                                       cv::Rect2f& mapped) const noexcept;

        private:
            struct InverseCoordinate {
                std::uint16_t x = 0;
                std::uint16_t y = 0;
            };
            
            [[nodiscard]] bool mapRgbLeftToRectifiedLeft(const cv::Point2f& point, cv::Point2f& mapped) const noexcept;

            std::uint32_t width_ = 0;
            std::uint32_t height_ = 0;

            // Dense nearest inverse of the calibration's rectified->source map.
            std::vector<cv::Point2f> rgb_left_to_rectified_;
            std::vector<std::uint8_t> valid_;

            bool initialized_ = false;
    };
}
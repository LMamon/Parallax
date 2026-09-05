#include <parallax/perception/image_space_mapper.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace parallax::perception {
    bool ImageSpaceMapper::initialize(const stereo::StereoCalibration& calibration) {
        if (!calibration.loaded()) return false;

        const auto& metadata = calibration.metadata();

        // StereoRectifier consumes these same maps directly. They describe,
        // for each rectified output pixel, which source RgbLeft coordinate
        // should be sampled. Object3D needs the opposite direction.
        return initialize(metadata.image_width, metadata.image_height, calibration.leftMapX(), calibration.leftMapY());
    }

    bool ImageSpaceMapper::initialize(std::uint32_t width, 
                                      std::uint32_t height,
                                      const std::vector<float>& rectified_to_rgb_x, 
                                      const std::vector<float>& rectified_to_rgb_y) {

        initialized_ = false;
        width_ = 0;
        height_ = 0;

        rgb_left_to_rectified_.clear();
        valid_.clear();

        if (width == 0 || height == 0) return false;

        // Inverse coordinates are stored as uint16_t to avoid retaining another
        // pair of full-resolution float maps. Current calibrated image dimensions
        // are far below the uint16_t coordinate limit.
        if (width > std::numeric_limits<std::uint16_t>::max() || height > std::numeric_limits<std::uint16_t>::max()) {
            return false;
        }

        const std::size_t elements = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

        if (rectified_to_rgb_x.size() != elements || rectified_to_rgb_y.size() != elements) {
            return false;
        }

        width_ = width;
        height_ = height;

        rgb_left_to_rectified_.resize(elements);
        valid_.assign(elements, 0);

        /*
         * Several rectified pixels can round onto the same RgbLeft source pixel.
         * Keep whichever rectified sample landed closest to that source-pixel
         * center. This makes inversion deterministic instead of depending on
         * traversal order.
         *
         * best_distance is initialization-only scratch storage and is released
         * before this function returns.
         */
        std::vector<float> best_distance(elements, std::numeric_limits<float>::infinity());

        /*
         * Existing calibration direction:
         *
         *     RectifiedLeft(x, y) -> RgbLeft(sample_x, sample_y)
         *
         * NanoOWL gives returns RgbLeft coordinates, build a nearest discrete
         * inverse once at startup rather than searching the full warp map for
         * every detection.
         */
        for (std::uint32_t rect_y = 0; rect_y < height_; ++rect_y) {
            for (std::uint32_t rect_x = 0; rect_x < width_; ++rect_x) {
                const std::size_t rect_index = static_cast<std::size_t>(rect_y) * width_ + rect_x;

                const float source_x = rectified_to_rgb_x[rect_index];
                const float source_y = rectified_to_rgb_y[rect_index];

                // Invalid/out-of-domain calibration samples never become valid
                // inverse entries. Consumers must handle an unmappable point
                // explicitly rather than silently receiving (0, 0).
                if (!std::isfinite(source_x) || !std::isfinite(source_y)) {
                    continue;
                }

                const auto source_x_round = static_cast<long>(std::lround(source_x));
                const auto source_y_round = static_cast<long>(std::lround(source_y));

                if (source_x_round < 0 ||
                    source_y_round < 0 ||
                    source_x_round >= static_cast<long>(width_) || source_y_round >= static_cast<long>(height_)) {
                    continue;
                }

                const float dx = source_x - static_cast<float>(source_x_round);
                const float dy = source_y - static_cast<float>(source_y_round);

                const float distance = dx * dx + dy * dy;
                const std::size_t source_index = static_cast<std::size_t>(source_y_round) * width_ + static_cast<std::size_t>(source_x_round);

                if (distance >= best_distance[source_index]) continue;

                best_distance[source_index] = distance;
                rgb_left_to_rectified_[source_index] = {static_cast<std::uint16_t>(rect_x), static_cast<std::uint16_t>(rect_y)};
                valid_[source_index] = 1;
            }
        }

        initialized_ = true;
        return true;
    }

    bool ImageSpaceMapper::mapPoint(const cv::Point2f& point, ImageSpace source, ImageSpace destination, cv::Point2f& mapped) const noexcept {

        if (!initialized_) return false;

        if (source == ImageSpace::Unknown || destination == ImageSpace::Unknown) {
            return false;
        }

        /*
         * Same-space mapping is intentionally explicit. This allows callers to
         * use one mapping API without treating an unknown or out-of-bounds point
         * as a valid identity transform.
         */
        if (source == destination) {
            if (!std::isfinite(point.x) ||
                !std::isfinite(point.y) ||
                point.x < 0.0F ||
                point.y < 0.0F ||
                point.x >= static_cast<float>(width_) || point.y >= static_cast<float>(height_)) {
                return false;
            }

            mapped = point;
            return true;
        }

        if (source == ImageSpace::RgbLeft && destination == ImageSpace::RectifiedLeft) {
            return mapRgbLeftToRectifiedLeft(point, mapped);
        }

        // only establishes the direction required by NanoOWL -> Depth.
        // Unsupported mappings must fail rather than implying that image spaces
        // are interchangeable.
        return false;
    }

    bool ImageSpaceMapper::mapRgbLeftToRectifiedLeft(const cv::Point2f& point, cv::Point2f& mapped) const noexcept {

        if (!std::isfinite(point.x) || !std::isfinite(point.y)) return false;

        /*
         * Object association ultimately samples a discrete depth image. Round
         * the semantic coordinate to its source pixel before consulting the
         * precomputed inverse calibration map.
         */
        const auto x = static_cast<long>(std::lround(point.x));
        const auto y = static_cast<long>(std::lround(point.y));

        if (x < 0 ||
            y < 0 ||
            x >= static_cast<long>(width_) || y >= static_cast<long>(height_)) {

            return false;
        }

        const std::size_t index = static_cast<std::size_t>(y) * width_ + static_cast<std::size_t>(x);
        /*
         * Distortion, crop boundaries, or the discrete inversion can leave
         * source pixels without a corresponding rectified sample.
         * association can reject or choose another supported sample later.
         */
        if (valid_[index] == 0) return false;

        const auto coordinate = rgb_left_to_rectified_[index];
        mapped = {static_cast<float>(coordinate.x), static_cast<float>(coordinate.y)};
        return true;
    }

    bool ImageSpaceMapper::mapRect(const cv::Rect2f& rect, ImageSpace source, ImageSpace destination, cv::Rect2f& mapped) const noexcept {
        if (!std::isfinite(rect.x) ||
            !std::isfinite(rect.y) ||
            !std::isfinite(rect.width) ||
            !std::isfinite(rect.height) ||
            rect.width <= 0.0F || rect.height <= 0.0F) {

            return false;
        }

        /*
         * Rectification is nonlinear, so a source rectangle is not guaranteed
         * to remain an exact rectangle. Mapping its four corners gives a
         * conservative rectified ROI for depth sampling; it is not a claim
         * about the object's physical shape.
         */
        const cv::Point2f corners[] = {{rect.x, rect.y},
                                       {rect.x + rect.width, rect.y},
                                       {rect.x, rect.y + rect.height},
                                       {rect.x + rect.width, rect.y + rect.height}};

        cv::Point2f mapped_corners[4];

        for (std::size_t i = 0; i < 4; ++i) {
            if (!mapPoint(corners[i], source, destination, mapped_corners[i])) {
                return false;
            }
        }

        float min_x = mapped_corners[0].x;
        float max_x = mapped_corners[0].x;
        float min_y = mapped_corners[0].y;
        float max_y = mapped_corners[0].y;

        for (std::size_t i = 1; i < 4; ++i) {
            min_x = std::min(min_x, mapped_corners[i].x);
            max_x = std::max(max_x, mapped_corners[i].x);
            min_y = std::min(min_y, mapped_corners[i].y);
            max_y = std::max(max_y, mapped_corners[i].y);
        }

        if (max_x <= min_x || max_y <= min_y) return false;
        mapped = {min_x, min_y, max_x - min_x, max_y - min_y};
        return true;
    }
}
#pragma once

#include <opencv2/core/types.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace parallax::perception {

    // CPU-facing NanoOWL detection result.
    // ProductMetadata owns source observation/timestamp provenance.
    struct DetectionSet {
        std::string query;
        std::uint64_t query_revision = 0;

        std::vector<cv::Rect2f> boxes;
        std::vector<float> scores;
        std::vector<std::int64_t> labels;

        [[nodiscard]] bool valid() const noexcept {
            return !query.empty() &&
                   query_revision != 0 &&
                   boxes.size() == scores.size() &&
                   scores.size() == labels.size();
        }

        [[nodiscard]] std::size_t size() const noexcept {
            return boxes.size();
        }

        [[nodiscard]] bool empty() const noexcept {
            return boxes.empty();
        }
    };
}
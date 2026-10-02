#pragma once

#include <parallax/core/product.hpp>
#include <parallax/perception/image_space.hpp>

#include <chrono>
#include <cstdint>
#include <string>

#include <opencv2/core/types.hpp>

namespace parallax::tracking {

    enum class TrackLifecycle : std::uint8_t {
        Idle = 0,
        Tentative,
        Tracking,
        Lost,
        Reacquiring
    };

    struct Track2D {
        std::uint64_t track_id = 0;

        std::string target_query;
        std::uint64_t target_revision = 0;

        cv::Rect2f box{};
        float quality = 0.0F;

        TrackLifecycle lifecycle = TrackLifecycle::Idle;
        perception::ImageSpace image_space = perception::ImageSpace::Unknown;

        core::SourceObservation source_observation{};
        core::SourceObservation last_detector_observation{};
        core::SourceObservation last_tracker_observation{};

        std::chrono::steady_clock::time_point last_detector_timestamp{};
        std::chrono::steady_clock::time_point last_tracker_timestamp{};

        [[nodiscard]] bool valid() const noexcept {
            return track_id != 0 &&
                   !target_query.empty() &&
                   target_revision != 0 &&
                   image_space != perception::ImageSpace::Unknown &&
                   lifecycle != TrackLifecycle::Idle;
        }
    };

}
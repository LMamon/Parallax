#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace parallax::localization {

    struct LocalizationPose {
        std::int64_t timestamp_ns = 0;
        std::uint64_t epoch = 0;

        std::array<float, 3> translation_m{};
        std::array<float, 4> rotation_xyzw{0.0F, 0.0F, 0.0F, 1.0F};
    };

    struct LocalizationOdometry {
        LocalizationPose pose{};
    };

    struct LocalizationTrajectory {
        std::uint64_t epoch = 0;
        std::vector<LocalizationPose> poses;
    };

    enum class LocalizationTrackingState : std::uint8_t {
        Uninitialized, Tracking, Lost
    };

    struct LocalizationState {
        LocalizationTrackingState tracking = LocalizationTrackingState::Uninitialized;

        std::uint64_t epoch = 0;
        std::uint64_t consumed_frames = 0;
        std::uint64_t input_gaps = 0;
        std::uint64_t session_resets = 0;
    };
}
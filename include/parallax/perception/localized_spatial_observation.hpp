#pragma once

#include <parallax/core/product.hpp>
#include <parallax/perception/object3d.hpp>

#include <chrono>
#include <cstdint>
#include <vector>

namespace parallax::perception {

    struct LocalizedSpatialObservation {
        Object3DSet objects;

        core::SourceObservation localization_observation{};
        std::uint64_t localization_epoch = 0;
        std::chrono::steady_clock::duration localization_time_delta{};

        [[nodiscard]] bool valid() const noexcept {
            return objects.valid() && localization_observation.valid();
        }
    };
}
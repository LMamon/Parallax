#pragma once

#include <parallax/camera/v4l2_device.hpp>
#include <parallax/isp/isp_config.hpp>

#include <array>
#include <cstdint>

namespace parallax::isp {

struct IspStatistics {
    std::array<std::uint32_t, 256> luminance_histogram{};
    std::uint64_t red_sum = 0;
    std::uint64_t green_sum = 0;
    std::uint64_t blue_sum = 0;
    std::uint64_t color_samples = 0;
    std::uint64_t total_samples = 0;
    std::uint64_t saturated_samples = 0;
    std::uint64_t sequence = 0;

    [[nodiscard]] bool valid() const noexcept { return total_samples != 0; }
};

struct AutoControlUpdate {
    std::int32_t exposure = 0;
    std::int32_t analogue_gain = 0;
    std::array<float, 3> white_balance{1.0F, 1.0F, 1.0F};
    bool exposure_changed = false;
    bool gain_changed = false;
    bool white_balance_changed = false;
};

class AutoController {
public:
    AutoController(IspConfig config,
                   parallax::camera::ControlRange exposure_range,
                   parallax::camera::ControlRange gain_range,
                   std::int32_t initial_exposure,
                   std::int32_t initial_gain);

    [[nodiscard]] AutoControlUpdate update(const IspStatistics& statistics);

private:
    [[nodiscard]] static float percentile(const IspStatistics& statistics, float fraction);
    [[nodiscard]] static std::int32_t quantize(std::int32_t value,
                                               const parallax::camera::ControlRange& range);

    IspConfig config_{};
    parallax::camera::ControlRange exposure_range_{};
    parallax::camera::ControlRange gain_range_{};
    std::int32_t exposure_ = 0;
    std::int32_t gain_ = 0;
    std::array<float, 3> white_balance_{1.0F, 1.0F, 1.0F};
};

} // namespace parallax::isp

#include <parallax/isp/auto_control.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace parallax::isp {
    namespace {

        std::int32_t scaledControl(std::int32_t current, float factor) {
            const double scaled = static_cast<double>(current) * static_cast<double>(factor);

            const double bounded = std::clamp(scaled,
                                            static_cast<double>(std::numeric_limits<std::int32_t>::min()),
                                            static_cast<double>(std::numeric_limits<std::int32_t>::max()));

            return static_cast<std::int32_t>(std::lround(bounded));
        }
    }

    AutoController::AutoController(IspConfig config,
                                parallax::camera::ControlRange exposure_range,
                                parallax::camera::ControlRange gain_range,
                                std::int32_t initial_exposure,
                                std::int32_t initial_gain)
        : config_(std::move(config)),
        exposure_range_(exposure_range),
        gain_range_(gain_range),
        exposure_(quantize(initial_exposure, exposure_range_)),
        gain_(quantize(initial_gain, gain_range_)),
        white_balance_(config_.white_balance) {}

    float AutoController::percentile(const IspStatistics& statistics, float fraction) {
        if (!statistics.valid()) return 0.0F;
        
        fraction = std::clamp(fraction, 0.0F, 1.0F);
        const std::uint64_t target = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(
            std::ceil(static_cast<double>(statistics.total_samples) * fraction)));

        std::uint64_t cumulative = 0;
        
        for (std::size_t i = 0; i < statistics.luminance_histogram.size(); ++i) {
            cumulative += statistics.luminance_histogram[i];
            if (cumulative >= target) return static_cast<float>(i) / 255.0F;
        }
        
        return 1.0F;
    }

    std::int32_t AutoController::quantize(std::int32_t value, const parallax::camera::ControlRange& range) {
        if (!range.valid()) return value;
        value = std::clamp(value, range.minimum, range.maximum);
        
        const std::int32_t step = std::max<std::int32_t>(range.step, 1);
        const std::int32_t offset = value - range.minimum;
        const std::int32_t snapped = range.minimum + ((offset + step / 2) / step) * step;
        
        return std::clamp(snapped, range.minimum, range.maximum);
    }

    AutoControlUpdate AutoController::update(const IspStatistics& statistics) {
        AutoControlUpdate result{};
        result.exposure = exposure_;
        result.analogue_gain = gain_;
        result.white_balance = white_balance_;

        if (!statistics.valid()) return result;

        if (config_.auto_exposure.enable && exposure_range_.valid() && gain_range_.valid()) {
            const float median = percentile(statistics, 0.50F);
            
            const float clipped = static_cast<float>(statistics.saturated_samples) /
                                static_cast<float>(statistics.total_samples);
            
            const float error = config_.auto_exposure.target_luma - median;
            
            const bool too_dark = error > config_.auto_exposure.deadband;
            const bool too_bright = error < -config_.auto_exposure.deadband ||
                                    clipped > config_.auto_exposure.highlight_fraction;

            const std::int32_t exposure_min = std::max(exposure_range_.minimum,
                                                    config_.auto_exposure.min_exposure_us);
            
            const std::int32_t exposure_max = std::min(exposure_range_.maximum,
                                                    config_.auto_exposure.max_exposure_us);

            const float safe_median = std::max(median, 1.0F / 255.0F);
            float correction = config_.auto_exposure.target_luma / safe_median;
            
            if (too_bright && clipped > config_.auto_exposure.highlight_fraction) {
                correction = std::min(correction, 0.85F);
            }
            correction = std::clamp(correction,
                                    1.0F / config_.auto_exposure.max_step_ratio,
                                    config_.auto_exposure.max_step_ratio);

            if (too_dark) {
                if (exposure_ < exposure_max) {
                    const auto next = quantize(std::clamp(scaledControl(exposure_, correction),
                                                        exposure_min, exposure_max),
                                            exposure_range_);
                    result.exposure_changed = next != exposure_;
                    exposure_ = next;
                } else if (gain_ < gain_range_.maximum) {
                    const auto next = quantize(scaledControl(gain_, correction), gain_range_);
                    result.gain_changed = next != gain_;
                    gain_ = next;
                }
            } else if (too_bright) {
                if (gain_ > gain_range_.minimum) {
                    const auto next = quantize(scaledControl(gain_, correction), gain_range_);
                    result.gain_changed = next != gain_;
                    gain_ = next;
                } else if (exposure_ > exposure_min) {
                    const auto next = quantize(std::clamp(scaledControl(exposure_, correction),
                                                        exposure_min, exposure_max),
                                            exposure_range_);
                    result.exposure_changed = next != exposure_;
                    exposure_ = next;
                }
            }
        }

        if (config_.auto_white_balance.enable && statistics.color_samples != 0 &&
            statistics.red_sum != 0 && statistics.green_sum != 0 && statistics.blue_sum != 0) {
            
            const float red = static_cast<float>(statistics.red_sum);
            const float green = static_cast<float>(statistics.green_sum);
            const float blue = static_cast<float>(statistics.blue_sum);
            
            const float desired_red = std::clamp(green / red,
                                                config_.auto_white_balance.min_gain,
                                                config_.auto_white_balance.max_gain);
            
            const float desired_blue = std::clamp(green / blue,
                                                config_.auto_white_balance.min_gain,
                                                config_.auto_white_balance.max_gain);

            const std::array<float, 3> desired{desired_red, 1.0F, desired_blue};
            
            const float alpha = config_.auto_white_balance.smoothing;
            auto next = white_balance_;
            
            for (std::size_t i = 0; i < next.size(); ++i) {
                next[i] = white_balance_[i] + alpha * (desired[i] - white_balance_[i]);
            }

            const float delta = std::fabs(next[0] - white_balance_[0]) +
                                std::fabs(next[1] - white_balance_[1]) +
                                std::fabs(next[2] - white_balance_[2]);
            
            if (delta > 1.0e-4F) {
                white_balance_ = next;
                result.white_balance_changed = true;
            }
        }

        result.exposure = exposure_;
        result.analogue_gain = gain_;
        result.white_balance = white_balance_;
        return result;
    }
}

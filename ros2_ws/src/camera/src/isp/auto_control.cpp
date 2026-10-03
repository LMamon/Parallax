#include <parallax/isp/auto_control.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace parallax::isp {
namespace {

    std::int32_t clampRound(double value,
                            std::int32_t minimum,
                            std::int32_t maximum) {
        return static_cast<std::int32_t>(
            std::lround(std::clamp(
                value,
                static_cast<double>(minimum),
                static_cast<double>(maximum))));
    }

}  // namespace

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

        const auto target = std::max<std::uint64_t>(1,
                                                    static_cast<std::uint64_t>(std::ceil(
                                                    static_cast<double>(statistics.total_samples) *fraction)));

        std::uint64_t cumulative = 0;

        for (std::size_t i = 0; i < statistics.luminance_histogram.size(); ++i) {
            cumulative += statistics.luminance_histogram[i];

            if (cumulative >= target) {
                return static_cast<float>(i) / 255.0F;
            }
        }

        return 1.0F;
    }

    std::int32_t AutoController::quantize(std::int32_t value, const parallax::camera::ControlRange& range) {

        if (!range.valid()) return value;

        value = std::clamp(value, range.minimum, range.maximum);

        const auto step = std::max<std::int32_t>(range.step, 1);

        const auto offset = value - range.minimum;
        const auto snapped = range.minimum + ((offset + step / 2) / step) * step;

        return std::clamp(snapped, range.minimum, range.maximum);
    }

    AutoControlUpdate AutoController::update(const IspStatistics& statistics) {
        AutoControlUpdate result{};

        result.exposure = exposure_;
        result.analogue_gain = gain_;
        result.white_balance = white_balance_;

        if (!statistics.valid()) return result;

        /*
        * SENSOR AUTO EXPOSURE / GAIN
        *
        * Exposure and analogue gain are treated as one sensor
        * brightness budget:
        *
        *     signal ~= exposure * gain
        *
        * Exposure is preferred because it avoids unnecessary gain,
        * but it may never exceed config.max_exposure. For the current
        * 3840x1200 @ 38 FPS mode that ceiling is 1500.
        *
        * Once exposure reaches that ceiling, remaining correction is
        * assigned to analogue gain.
        *
        * Both controls may change from one statistics sample.
        */
        if (config_.auto_exposure.enable && exposure_range_.valid() && gain_range_.valid()) {

            const float median = percentile(statistics, 0.50F);

            const float clipped = static_cast<float>(statistics.saturated_samples) / static_cast<float>(statistics.total_samples);

            const float error = config_.auto_exposure.target_luma - median;
            const bool too_dark = error > config_.auto_exposure.deadband;
            const bool too_bright = error <-config_.auto_exposure.deadband || clipped > config_.auto_exposure.highlight_fraction;

            if (too_dark || too_bright) {
                const auto exposure_min = quantize(std::max(exposure_range_.minimum,
                                                            config_.auto_exposure.min_exposure),
                                                            exposure_range_);

                const auto exposure_max = quantize(std::min(exposure_range_.maximum,
                                                            config_.auto_exposure.max_exposure),
                                                            exposure_range_);

                const auto gain_min = gain_range_.minimum;
                const auto gain_max = gain_range_.maximum;
                const float safe_luma = std::max(median, 1.0F / 255.0F);

                /*
                * Calculate the required correction directly.
                *
                * max_step_ratio remains useful as a stability limit,
                * but rather than the old 6% crawl, allow several
                * configured steps per statistics update.
                *
                * With max_step_ratio=1.06:
                *
                *     1.06^4 ~= 1.26
                *
                * so one 10 Hz update can correct roughly 26%.
                */
                const float max_correction = std::pow(config_.auto_exposure.max_step_ratio, 4.0F);

                float correction = config_.auto_exposure.target_luma / safe_luma;

                if (too_bright && clipped > config_.auto_exposure.highlight_fraction) {
                    correction = std::min(correction, 0.75F);
                }

                correction = std::clamp(correction, 1.0F / max_correction, max_correction);

                /*
                * Desired total sensor signal.
                *
                * Gain values are driver-native controls, so we don't
                * claim they are linear physical gain. This controller
                * uses them monotonically as the sensor exposes them.
                */
                const double current_signal = static_cast<double>(exposure_) * static_cast<double>(gain_);

                const double desired_signal = current_signal * static_cast<double>(correction);

                std::int32_t next_exposure = exposure_;
                std::int32_t next_gain = gain_;

                if (too_dark) {
                    /*
                    * Brightening:
                    *
                    * Prefer exposure until the 38 FPS ceiling.
                    * Put any remaining requirement into gain.
                    */
                    const double desired_exposure = desired_signal / static_cast<double>(gain_);

                    next_exposure = quantize(clampRound(desired_exposure,
                                                        exposure_min,
                                                        exposure_max),
                                                        exposure_range_);

                    const double required_gain = desired_signal / static_cast<double>(std::max(next_exposure, 1));

                    next_gain = quantize(clampRound(required_gain,
                                                    gain_min,
                                                    gain_max),
                                                    gain_range_);
                } else {
                    /*
                    * Darkening:
                    *
                    * Remove analogue gain first. If minimum gain
                    * isn't enough, shorten exposure.
                    */
                    const double desired_gain = desired_signal / static_cast<double>(std::max(exposure_, 1));

                    next_gain = quantize(clampRound(desired_gain,
                                                    gain_min,
                                                    gain_max),
                                                    gain_range_);

                    const double required_exposure = desired_signal / static_cast<double>(std::max(next_gain, 1));

                    next_exposure = quantize(clampRound(required_exposure,
                                                        exposure_min,
                                                        exposure_max),
                                                        exposure_range_);
                }

                if (next_exposure != exposure_) {
                    exposure_ = next_exposure;
                    result.exposure_changed = true;
                }

                if (next_gain != gain_) {
                    gain_ = next_gain;
                    result.gain_changed = true;
                }
            }
        }

        /*
        * ISP AUTO WHITE BALANCE
        *
        * This remains independent of sensor AE/AGC.
        */
        if (config_.auto_white_balance.enable &&
            statistics.color_samples != 0 &&
            statistics.red_sum != 0 &&
            statistics.green_sum != 0 &&
            statistics.blue_sum != 0) {

            const float red = static_cast<float>(statistics.red_sum);
            const float green = static_cast<float>(statistics.green_sum);
            const float blue = static_cast<float>(statistics.blue_sum);

            const float desired_red = std::clamp(green / red,
                                                 config_.auto_white_balance.min_gain,
                                                 config_.auto_white_balance.max_gain);

            const float desired_blue = std::clamp(green / blue,
                                                  config_.auto_white_balance.min_gain,
                                                  config_.auto_white_balance.max_gain);

            const std::array<float, 3> desired {
                desired_red, 1.0F, desired_blue
            };

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
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>

namespace parallax::isp {

    struct StatisticsConfig {
        std::uint32_t sample_stride = 8;
        float update_hz = 8.0F;
    };

    struct AutoExposureConfig {
        bool enable = true;
        std::int32_t min_exposure_us = 100;
        std::int32_t max_exposure_us = 8000;
        float target_luma = 0.45F;
        float deadband = 0.04F;
        float max_step_ratio = 1.25F;
        float highlight_fraction = 0.02F;
    };

    struct AutoWhiteBalanceConfig {
        bool enable = true;
        float smoothing = 0.08F;
        float min_gain = 0.5F;
        float max_gain = 3.0F;
    };

    struct IspConfig {
        bool enable = true;
        std::uint16_t black_level = 64;
        
        float gamma = 2.2F;
        
        std::array<float, 3> white_balance{2.0F, 1.0F, 1.8F};
        std::array<float, 9> color_matrix{
            1.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 1.0F,
        };

        StatisticsConfig statistics{};
        
        AutoExposureConfig auto_exposure{};
        AutoWhiteBalanceConfig auto_white_balance{};

        bool loadFromFile(const std::filesystem::path& path);
    };

}

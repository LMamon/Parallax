#include <parallax/isp/isp_config.hpp>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <iostream>

namespace parallax::isp {
    namespace {

        bool finitePositive(float value) {
            return std::isfinite(value) && value > 0.0F;
        }

        bool loadMatrix(const YAML::Node& node, std::array<float, 9>& matrix) {
            if (!node || !node.IsSequence()) return false;

            if (node.size() == 9) {
                for (std::size_t i = 0; i < 9; ++i) matrix[i] = node[i].as<float>();
                return true;
            }

            if (node.size() == 3) {
                for (std::size_t row = 0; row < 3; ++row) {
                    if (!node[row].IsSequence() || node[row].size() != 3) return false;
                    for (std::size_t col = 0; col < 3; ++col) {
                        matrix[row * 3 + col] = node[row][col].as<float>();
                    }
                }
                return true;
            }

            return false;
        }

    }

    bool IspConfig::loadFromFile(const std::filesystem::path& path) {
        try {
            const YAML::Node root = YAML::LoadFile(path.string());

            if (root["enable"]) enable = root["enable"].as<bool>();
            if (root["black_level"]) black_level = root["black_level"].as<std::uint16_t>();
            if (root["gamma"]) gamma = root["gamma"].as<float>();

            if (const auto wb = root["white_balance"]) {
                if (wb["red"]) white_balance[0] = wb["red"].as<float>();
                if (wb["green"]) white_balance[1] = wb["green"].as<float>();
                if (wb["blue"]) white_balance[2] = wb["blue"].as<float>();
            }

            if (const auto correction = root["color_correction"]) {
                if (const auto matrix = correction["matrix"]) {
                    if (!loadMatrix(matrix, color_matrix)) {
                        std::cerr << "ISP config: color_correction.matrix must contain 9 values or 3 rows of 3\n";
                        return false;
                    }
                }
            }

            if (const auto stats = root["statistics"]) {
                if (stats["sample_stride"]) statistics.sample_stride = stats["sample_stride"].as<std::uint32_t>();
                if (stats["update_hz"]) statistics.update_hz = stats["update_hz"].as<float>();
            }

            if (const auto ae = root["auto_exposure"]) {
                if (ae["enable"]) auto_exposure.enable = ae["enable"].as<bool>();
                if (ae["min_exposure_us"]) auto_exposure.min_exposure_us = ae["min_exposure_us"].as<std::int32_t>();
                if (ae["max_exposure_us"]) auto_exposure.max_exposure_us = ae["max_exposure_us"].as<std::int32_t>();
                if (ae["target_luma"]) auto_exposure.target_luma = ae["target_luma"].as<float>();
                if (ae["deadband"]) auto_exposure.deadband = ae["deadband"].as<float>();
                if (ae["max_step_ratio"]) auto_exposure.max_step_ratio = ae["max_step_ratio"].as<float>();
                if (ae["highlight_fraction"]) auto_exposure.highlight_fraction = ae["highlight_fraction"].as<float>();
            }

            if (const auto awb = root["auto_white_balance"]) {
                if (awb["enable"]) auto_white_balance.enable = awb["enable"].as<bool>();
                if (awb["smoothing"]) auto_white_balance.smoothing = awb["smoothing"].as<float>();
                if (awb["min_gain"]) auto_white_balance.min_gain = awb["min_gain"].as<float>();
                if (awb["max_gain"]) auto_white_balance.max_gain = awb["max_gain"].as<float>();
            }
        } catch (const YAML::Exception& error) {
            std::cerr << "ISP config: " << error.what() << '\n';
            return false;
        }

        if (black_level >= 1023 || !finitePositive(gamma)) return false;
        if (statistics.sample_stride == 0 || !finitePositive(statistics.update_hz)) return false;

        if (auto_exposure.min_exposure_us <= 0 ||
            auto_exposure.max_exposure_us < auto_exposure.min_exposure_us ||
            !(auto_exposure.target_luma > 0.0F && auto_exposure.target_luma < 1.0F) ||
            !(auto_exposure.deadband >= 0.0F && auto_exposure.deadband < 1.0F) ||
            auto_exposure.max_step_ratio < 1.0F ||
            !(auto_exposure.highlight_fraction >= 0.0F && auto_exposure.highlight_fraction <= 1.0F)) return false;

        if (!(auto_white_balance.smoothing > 0.0F && auto_white_balance.smoothing <= 1.0F) ||
            !finitePositive(auto_white_balance.min_gain) ||
            auto_white_balance.max_gain < auto_white_balance.min_gain) return false;

        for (float gain : white_balance) if (!finitePositive(gain)) return false;
        for (float value : color_matrix) if (!std::isfinite(value)) return false;

        return true;
    }

} 

#include <parallax/stereo/calibration.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <filesystem>
#include <iostream>

namespace parallax::stereo {

namespace {

bool loadBinaryMap(const std::filesystem::path& path, std::vector<float>& map, std::size_t expected_elements) {
    const std::uintmax_t expected_bytes = expected_elements * sizeof(float);

    std::error_code ec;
    const std::uintmax_t actual_bytes = std::filesystem::file_size(path, ec);

    if (ec) {
        std::cerr << "Failed to read calibration map size: " << path << '\n';
        return false;
    }

    if (actual_bytes != expected_bytes) {
        std::cerr << "Invalid calibration map size: " << path
                  << "\nExpected: " << expected_bytes
                  << " bytes\nActual: " << actual_bytes << " bytes\n";
        return false;
    }

    map.resize(expected_elements);

    std::ifstream file(path, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open calibration map: " << path << '\n';
        return false;
    }

    file.read(reinterpret_cast<char*>(map.data()), static_cast<std::streamsize>(expected_bytes));

    if (!file) {
        std::cerr << "Failed to read calibration map: " << path << '\n';
        return false;
    }

    return true;
}

template <std::size_t N>
bool loadMatrix(const nlohmann::json& json, const char* name, std::array<double, N>& output) {
    if (!json.contains(name) || !json[name].is_array()) {
        std::cerr << "Missing calibration matrix: " << name << '\n';
        return false;
    }

    std::size_t index = 0;

    for (const auto& row : json[name]) {
        if (!row.is_array()) {
            std::cerr << "Invalid calibration matrix: " << name << '\n';
            return false;
        }

        for (const auto& value : row) {
            if (index >= N || !value.is_number()) {
                std::cerr << "Invalid calibration matrix: " << name << '\n';
                return false;
            }

            output[index++] = value.get<double>();
        }
    }

    if (index != N) {
        std::cerr << "Invalid calibration matrix element count: " << name
                  << "\nExpected: " << N
                  << "\nActual: " << index << '\n';
        return false;
    }

    return true;
}

}

bool StereoCalibration::load(const std::filesystem::path& directory) {
    loaded_ = false;

    const auto json_path = directory / "calibration.json";

    std::ifstream file(json_path);

    if (!file) {
        std::cerr << "Failed to open calibration file: " << json_path << '\n';
        return false;
    }

    nlohmann::json json;

    try {
        file >> json;

        metadata_.image_width = json.at("image_width").get<std::uint32_t>();
        metadata_.image_height = json.at("image_height").get<std::uint32_t>();

        metadata_.virtual_fx = json.at("virtual_fx").get<double>();
        metadata_.virtual_fy = json.at("virtual_fy").get<double>();

        metadata_.virtual_cx = json.at("virtual_cx").get<double>();
        metadata_.virtual_cy = json.at("virtual_cy").get<double>();

        metadata_.rectification_alpha = json.at("rectification_alpha").get<double>();
        metadata_.baseline_mm = json.at("baseline_mm").get<double>();
        metadata_.extrinsics_convention = json.at("extrinsics_convention").get<std::string>();

    } catch (const nlohmann::json::exception& e) {
        std::cerr << "Invalid calibration JSON: " << e.what() << '\n';
        return false;
    }

    if (!loadMatrix(json, "P1", p1_) ||
        !loadMatrix(json, "P2", p2_) ||
        !loadMatrix(json, "Q", q_) ||
        !loadMatrix(json, "R1", r1_) ||
        !loadMatrix(json, "R2", r2_)) {
        return false;
    }

    if (metadata_.image_width == 0 || metadata_.image_height == 0) {
        std::cerr << "Invalid calibration image dimensions\n";
        return false;
    }

    const std::size_t map_elements = static_cast<std::size_t>(metadata_.image_width) * metadata_.image_height;

    if (!loadBinaryMap(directory / "left_map_x.bin", left_map_x_, map_elements) ||
        !loadBinaryMap(directory / "left_map_y.bin", left_map_y_, map_elements) ||
        !loadBinaryMap(directory / "right_map_x.bin", right_map_x_, map_elements) ||
        !loadBinaryMap(directory / "right_map_y.bin", right_map_y_, map_elements)) {

        return false;
    }

    loaded_ = true;
    return true;
}

}
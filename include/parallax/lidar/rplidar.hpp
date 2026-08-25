#pragma once

#include <parallax/lidar/frame_types.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace sl {
    class ILidarDriver;
    class IChannel;
}

namespace parallax::lidar {

    /**
     * Thin Parallax adapter around the SLAMTEC SDK
     *
     * This class owns the device connection and scan lifecycle. fixed-point
     * measurements stop at this boundary; callers receive ordinary
     * Parallax LidarScan values in radians and meters.
     */
    class Rplidar {
    public:
        Rplidar() = default;
        ~Rplidar();

        Rplidar(const Rplidar&) = delete;
        Rplidar& operator=(const Rplidar&) = delete;

        bool initialize(const std::string& device = "/dev/ttyUSB0", std::uint32_t baud_rate = 460800);

        bool capture(LidarScan& scan);
        void shutdown() noexcept;

        [[nodiscard]] bool initialized() const noexcept {
            return initialized_;
        }

    private:
        sl::ILidarDriver* driver_{nullptr};
        sl::IChannel* channel_{nullptr};

        bool initialized_{false};
        bool scanning_{false};
    };
}
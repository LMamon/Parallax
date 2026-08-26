#pragma once

#include <parallax/camera/v4l2_device.hpp>

#include <cstdint>
#include <string>

#include <linux/ioctl.h>
#include <linux/videodev2.h>

namespace parallax::camera {

    struct ArducamI2C {
        std::uint16_t reg = 0;
        std::uint16_t val = 0;
    };

    struct ArducamDeviceRegister {
        std::uint16_t reg = 0;
        std::uint16_t val = 0;
    };

    // Arducam private V4L2 ioctls.
    constexpr auto VIDIOC_R_I2C = _IOWR('V', BASE_VIDIOC_PRIVATE + 0, ArducamI2C);
    constexpr auto VIDIOC_W_I2C = _IOWR('V', BASE_VIDIOC_PRIVATE + 1, ArducamI2C);
    constexpr auto VIDIOC_R_DEV = _IOWR('V', BASE_VIDIOC_PRIVATE + 2, ArducamDeviceRegister);
    constexpr auto VIDIOC_W_DEV = _IOWR('V', BASE_VIDIOC_PRIVATE + 3, ArducamDeviceRegister);

    class ArducamDevice : public V4L2Device {
        public:
            explicit ArducamDevice(const std::string& device);

            [[nodiscard]]std::uint16_t readSensorRegister(std::uint16_t reg) const;

            bool writeSensorRegister(std::uint16_t reg, std::uint16_t value) const;

            [[nodiscard]] std::uint32_t readControllerRegister(std::uint16_t reg) const;

            bool writeControllerRegister(std::uint16_t reg, std::uint32_t value) const;
    };

} // namespace parallax::camera
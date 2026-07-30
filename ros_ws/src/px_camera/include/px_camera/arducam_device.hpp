#pragma once

#include <px_camera/v4l2_device.hpp>
#include <cstdint>
#include <string>

#include <linux/ioctl.h>
#include <linux/videodev2.h>

namespace px_camera {
    
struct ArducamI2C {
    std::uint16_t reg;
    std::uint16_t val;
};

struct ArducamDev {
    std::uint16_t reg;
    std::uint16_t val;
};

//arducam private V4L2 ioctl commands.
#define VIDIOC_R_I2C _IOWR('V', BASE_VIDIOC_PRIVATE + 0, ArducamI2C)
#define VIDIOC_W_I2C _IOWR('V', BASE_VIDIOC_PRIVATE + 1, ArducamI2C)

#define VIDIOC_R_DEV _IOWR('V', BASE_VIDIOC_PRIVATE + 2, ArducamDev)
#define VIDIOC_W_DEV _IOWR('V', BASE_VIDIOC_PRIVATE + 3, ArducamDev)

class ArducamDevice : public V4L2Device {
    public:
        explicit ArducamDevice(const std::string& device);

        // AR0234 image-sensor registers
        std::uint16_t readSensor(std::uint16_t reg) const;
        bool writeSensor(std::uint16_t reg, std::uint16_t value) const;

        // Arducam controller/firmware registers
        std::uint32_t readDevice(std::uint16_t reg) const;
        bool writeDevice(std::uint16_t reg, std::uint32_t value) const;
};

}
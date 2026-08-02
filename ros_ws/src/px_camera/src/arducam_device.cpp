#include "px_camera/arducam_device.hpp"
#include <px_camera/v4l2_device.hpp>
#include <px_camera/logger.hpp>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <sys/ioctl.h>

namespace px_camera {
    ArducamDevice::ArducamDevice(const std::string& device) : V4L2Device(device) {}
    
    std::uint16_t ArducamDevice::readSensor(std::uint16_t reg) const {
        if (!isOpen()) {
            logMessage("Can't read sensor register: device is not open");
            return 0;
        }

        //arducam's driver uses a private ioctl to access sensor I2C registers.
        ArducamI2C data{};
        data.reg = reg;

        if (::ioctl(fileDescriptor(), VIDIOC_R_I2C, &data) < 0) {
            logError("VIDIOC_R_I2C");
            return 0;
        }
        return data.val;
    }

    bool ArducamDevice::writeSensor(std::uint16_t reg, std::uint16_t value) const {
        if (!isOpen()) {
            logMessage("Can't write sensor register: device is not open");
            return false;
        }

        ArducamI2C data{};
        data.reg = reg;
        data.val = value;

        if (::ioctl(fileDescriptor(), VIDIOC_W_I2C, &data) < 0) {
            logError("VIDIOC_W_I2C");
            return false;
        }
        return true;
    }

    std::uint32_t ArducamDevice::readDevice(std::uint16_t reg) const {
        if (!isOpen()) {
            logMessage("Can't read Arducam register: device is not open");
            return 0;
        }
        //device registers belong to the Arducam controller/firmware,
        //not the AR0234 image sensor itself.
        ArducamDev data{};
        data.reg = reg;

        if (::ioctl(fileDescriptor(), VIDIOC_R_DEV, &data) < 0) {
            logError("VIDIOC_R_DEV");
            return 0;
        }
        return data.val;
    }

    bool ArducamDevice::writeDevice(std::uint16_t reg, std::uint32_t value) const {
        if (!isOpen()) {
            logMessage("Can't write Arducam register: device is not open");
            return false;
        }

        ArducamDev data{};
        data.reg = reg;
        data.val = value;

        if (::ioctl(fileDescriptor(), VIDIOC_W_DEV, &data) < 0) {
            logError("VIDIOC_W_DEV");
            return false;
        }
        return true;
    }
}

#include "px_camera/arducam_device.hpp"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <sys/ioctl.h>

namespace px_camera {
    ArducamDevice::ArducamDevice(const std::string& device) : V4L2Device(device) {}
    
    std::uint16_t ArducamDevice::readSensor(std::uint16_t reg) const {
        if (!isOpen()) {
            std::cerr << "Cant read sensor register: device is not open\n";
            return 0;
        }

        //arducam's driver uses a private ioctl to access sensor I2C registers.
        ArducamI2C data{};
        data.reg = reg;

        if (::ioctl(fd_, VIDIOC_R_I2C, &data) < 0) {
            std::cerr << "Sensor register read failed: " << std::strerror(errno) << "\n";
            return 0;
        }
        return data.val;
    }

    bool ArducamDevice::writeSensor(std::uint16_t reg, std::uint16_t value) const {
        if (!isOpen()) {
            std::cerr << "Cant write sensor register: device is not open\n";
            return false;
        }

        ArducamI2C data{};
        data.reg = reg;
        data.val = value;

        if (::ioctl(fd_, VIDIOC_W_I2C, &data) < 0) {
            std::cerr << "Sensor register write failed: " << std::strerror(errno) << '\n';
            return false;
        }
        return true;
    }

    std::uint32_t ArducamDevice::readDevice(std::uint16_t reg) const {
        if (!isOpen()) {
            std::cerr << "Cant read Arducam register: device is not open\n";
            return 0;
        }
        //device registers belong to the Arducam controller/firmware,
        //not the AR0234 image sensor itself.
        ArducamDev data{};
        data.reg = reg;

        if (::ioctl(fd_, VIDIOC_R_DEV, &data) < 0) {
            std::cerr << "Arducam register read failed: " << std::strerror(errno) << "\n";
            return 0;
        }
        return data.val;
    }

    bool ArducamDevice::writeDevice(std::uint16_t reg, std::uint32_t value) const {
        if (!isOpen()) {
            std::cerr << "Cant write Arducam register: device is not open\n";
            return false;
        }

        ArducamDev data{};
        data.reg = reg;
        data.val = value;

        if (::ioctl(fd_, VIDIOC_W_DEV, &data) < 0) {
            std::cerr << "Arducam register write failed: " << std::strerror(errno) << "\n";
            return false;
        }
        return true;
    }
}

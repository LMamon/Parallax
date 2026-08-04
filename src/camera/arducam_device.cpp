#include <parallax/camera/arducam_device.hpp>

#include <parallax/camera/logger.hpp>

#include <sys/ioctl.h>

namespace parallax::camera {

    ArducamDevice::ArducamDevice(const std::string& device) : V4L2Device(device) {}

    std::uint16_t ArducamDevice::readSensorRegister(std::uint16_t reg) const {
        if (!isOpen()) {
            logMessage("readSensor: device is not open");
            return 0;
        }

        ArducamI2C data{};
        data.reg = reg;

        if (::ioctl(fileDescriptor(), VIDIOC_R_I2C, &data) < 0) {
            logError("VIDIOC_R_I2C");
            return 0;
        }

        return data.val;
    }

    bool ArducamDevice::writeSensorRegister(std::uint16_t reg, std::uint16_t value) const {
        if (!isOpen()) {
            logMessage("writeSensor: device is not open");
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

    std::uint32_t ArducamDevice::readControllerRegister(std::uint16_t reg) const {
        if (!isOpen()) {
            logMessage("readDevice: device is not open");
            return 0;
        }

        ArducamDeviceRegister data{};
        data.reg = reg;

        if (::ioctl(fileDescriptor(), VIDIOC_R_DEV, &data) < 0) {
            logError("VIDIOC_R_DEV");
            return 0;
        }

        return data.val;
    }

    bool ArducamDevice::writeControllerRegister(std::uint16_t reg, std::uint32_t value) const {
        if (!isOpen()) {
            logMessage("writeDevice: device is not open");
            return false;
        }

        ArducamDeviceRegister data{};
        data.reg = reg;
        data.val = value;

        if (::ioctl(fileDescriptor(), VIDIOC_W_DEV, &data) < 0) {
            logError("VIDIOC_W_DEV");
            return false;
        }

        return true;
    }

} // namespace parallax::camera
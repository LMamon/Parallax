#include <parallax/lidar/rplidar.hpp>

#include <sl_lidar.h>
#include <sl_lidar_driver.h>

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <limits>

namespace parallax::lidar {

    namespace {
        constexpr std::size_t kMaxScanNodes = 8192;
        constexpr float kPi = 3.14159265358979323846f;
        constexpr float kDegreesToRadians = kPi / 180.0f;

        /**
         * SLAMTEC HQ nodes store:
         *
         *   angle -> Q14 representation where 16384 == 90 degrees
         *   distance -> Q2 millimeters
         *
         * Normalize units so they never leak into the rest of
         * Parallax.
         */
        float angleRadians(const sl_lidar_response_measurement_node_hq_t& node) {
            const float angle_degrees = static_cast<float>(node.angle_z_q14) * 90.0f / 16384.0f;

            return angle_degrees * kDegreesToRadians;
        }

        float rangeMeters(const sl_lidar_response_measurement_node_hq_t& node) {
                          const float range_mm = static_cast<float>(node.dist_mm_q2) / 4.0f;

            return range_mm / 1000.0f;
        }
    }

    Rplidar::~Rplidar() { shutdown(); }

    bool Rplidar::initialize(const std::string& device, std::uint32_t baud_rate) {
        if (initialized_) return true;

        auto driver_result = sl::createLidarDriver();
        if (!driver_result) {
            std::cerr << "Rplidar: failed to create SLAMTEC driver\n";
            return false;
        }

        driver_ = *driver_result;

        auto channel_result = sl::createSerialPortChannel(device, baud_rate);
        if (!channel_result) {
            std::cerr << "Rplidar: failed to create serial channel for " << device << '\n';

            shutdown();
            return false;
        }

        channel_ = *channel_result;

        const sl_result connect_result = driver_->connect(channel_);
        if (SL_IS_FAIL(connect_result)) {
            std::cerr << "Rplidar: failed to connect to " << device
                      << " at "<< baud_rate
                      << " baud\n";

            shutdown();
            return false;
        }

        sl_lidar_response_device_info_t device_info{};
        const sl_result info_result = driver_->getDeviceInfo(device_info);
        if (SL_IS_FAIL(info_result)) {
            std::cerr << "Rplidar: failed to retrieve device information\n";

            shutdown();
            return false;
        }

        sl_lidar_response_device_health_t health{};
        const sl_result health_result = driver_->getHealth(health);
        if (SL_IS_FAIL(health_result)) {
            std::cerr << "Rplidar: failed to retrieve device health\n";

            shutdown();
            return false;
        }

        if (health.status == SL_LIDAR_STATUS_ERROR) {
            std::cerr << "Rplidar: device reported internal error "
                      << health.error_code << '\n';

            shutdown();
            return false;
        }

        /**
         * The C1 controls its scan motor through the SDK. Use the SDK-selected
         * default speed rather than embedding a model-specific RPM here.
         */
        const sl_result motor_result = driver_->setMotorSpeed();
        if (SL_IS_FAIL(motor_result)) {
            std::cerr << "Rplidar: failed to start scan motor\n";
            shutdown();
            return false;
        }

        const sl_result scan_result = driver_->startScan(false, true);
        if (SL_IS_FAIL(scan_result)) {
            std::cerr << "Rplidar: failed to start scan\n";
            shutdown();
            return false;
        }

        scanning_ = true;

        /**
         * The C1 does not provide a complete scan immediately after startScan().
         * Allow the motor/scanner to reach its operating state before the first
         * acquisition, matching the startup behavior required by SLAMTEC's working
         * simple_grabber example.
         *
         * This delay belongs at initialization, not in capture(), so steady-state
         * scan acquisition does not pay the startup cost repeatedly.
         */
        std::this_thread::sleep_for(std::chrono::seconds{3});
        initialized_ = true;

        std::cout << "RPLIDAR: connected " << device
                  << " @ " << baud_rate
                  << " baud | firmware " << (device_info.firmware_version >> 8) << '.';

        const auto firmware_minor = device_info.firmware_version & 0xFF;
        if (firmware_minor < 10) {
            std::cout << '0';
        }

        std::cout << firmware_minor << " | hardware "
                  << static_cast<int>(device_info.hardware_version) << '\n';

        return true;
    }


    bool Rplidar::capture(LidarScan& scan) {
        if (!initialized_ || driver_ == nullptr) {
            return false;
        }

        std::array<sl_lidar_response_measurement_node_hq_t, kMaxScanNodes> nodes{};
        std::size_t count = nodes.size();

        const sl_result grab_result = driver_->grabScanDataHq(nodes.data(), count);
        if (SL_IS_FAIL(grab_result)) {
            std::cerr << "Rplidar: failed to acquire scan, code 0x"
                      << std::hex << grab_result
                      << std::dec << '\n';

            return false;
        }

        const sl_result ascend_result = driver_->ascendScanData(nodes.data(), count);
        if (SL_IS_FAIL(ascend_result)) {
            std::cerr << "Rplidar: failed to order scan measurements\n";

            return false;
        }

        scan.points.clear();
        scan.points.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            const auto& node = nodes[i];
            const float range_m = rangeMeters(node);
            LidarPoint point;
            point.angle_rad = angleRadians(node);

            /**
            * Preserve every angular slot returned by ascendScanData().
            *
            * SLAMTEC uses zero distance for no-return measurements, but
            * ascendScanData() assigns those nodes an angle based on the scan's
            * 360 / count increment before sorting the full revolution. Removing
            * them here would destroy that regular angular representation and make
            * the scan unsuitable for Foxglove LaserScan.
            */

            point.valid = std::isfinite(range_m) && range_m > 0.0f;
            point.range_m = point.valid ? range_m : std::numeric_limits<float>::quiet_NaN();

            point.quality = static_cast<std::uint8_t>(node.quality >>SL_LIDAR_RESP_MEASUREMENT_QUALITY_SHIFT);
            scan.points.push_back(point);
        }
        return scan.valid();
    }


    void Rplidar::shutdown() noexcept {
        if (driver_ != nullptr) {
            if (scanning_) {
                driver_->stop();
                driver_->setMotorSpeed(0);
            }
            driver_->disconnect();
        }

        /**
         * Driver/channel instances are created by SDK factory functions.
         * The SDK owns the concrete implementation types; delete through their
         * public interfaces after the device has been stopped/disconnected.
         */
        delete driver_;
        delete channel_;

        driver_ = nullptr;
        channel_ = nullptr;

        scanning_ = false;
        initialized_ = false;
    }
}
#include <parallax/lidar/rplidar.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>

int main() {
    parallax::lidar::Rplidar lidar;
    if (!lidar.initialize()) {
        std::cerr << "RPLIDAR smoke: initialization failed\n";
        return 1;
    }

    constexpr int kScanCount = 5;
    for (int i = 0; i < kScanCount; ++i) {
        parallax::lidar::LidarScan scan;

        if (!lidar.capture(scan)) {
            std::cerr << "RPLIDAR smoke: scan " << i << " acquisition failed\n";
            return 1;
        }

        if (!scan.valid()) {
            std::cerr << "RPLIDAR smoke: scan " << i << " contained no valid points\n";
            return 1;
        }

        const auto nearest = std::min_element(scan.points.begin(),
                                              scan.points.end(),
                                              [](const auto& lhs, const auto& rhs) {
                    
                    return lhs.range_m < rhs.range_m;
                });

        const auto farthest = std::max_element(scan.points.begin(),
                                               scan.points.end(),
                                               [](const auto& lhs, const auto& rhs) {
                    
                    return lhs.range_m < rhs.range_m;
                });

        std::cout << std::fixed << std::setprecision(3)
                  << "RPLIDAR smoke: scan=" << i
                  << " | points=" << scan.points.size()
                  << " | nearest=" << nearest->range_m << " m"
                  << " | farthest=" << farthest->range_m << " m"
                  << " | first_angle=" << scan.points.front().angle_rad << " rad"
                  << " | last_angle=" << scan.points.back().angle_rad << " rad"
                  << '\n';
    }

    lidar.shutdown();
    return 0;
}
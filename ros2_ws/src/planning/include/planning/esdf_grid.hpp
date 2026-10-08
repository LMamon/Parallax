#pragma once

#include <cstddef>
#include <vector>

#include <nvblox_msgs/srv/esdf_and_gradients.hpp>

namespace planning {

class EsdfGrid {
    public:
        bool update(const nvblox_msgs::srv::EsdfAndGradients::Response &response);

        bool valid() const;

        float distanceAt(double x, double y, double z) const;

        bool isStateValid(double x,
                          double y,
                          double z,
                          double clearance_m) const;

        double voxelSize() const { return voxel_size_m_; }

    private:
        double origin_x_{0.0};
        double origin_y_{0.0};
        double origin_z_{0.0};

        double voxel_size_m_{0.0};

        std::size_t size_x_{0};
        std::size_t size_y_{0};
        std::size_t size_z_{0};

        std::size_t stride_y_{0};
        std::size_t stride_z_{0};

        std::vector<float> data_;
    };

}

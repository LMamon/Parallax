#include "planning/esdf_grid.hpp"

#include <cmath>
#include <limits>

namespace planning {

    bool EsdfGrid::update(
        const nvblox_msgs::srv::EsdfAndGradients::Response &response) {

        const auto &array = response.esdf_and_gradients;

        if (array.layout.dim.size() != 3) return false;

        if (array.layout.dim[0].label != "x" ||
            array.layout.dim[1].label != "y" ||
            array.layout.dim[2].label != "z") {
            return false;
        }

        origin_x_ = response.origin_m.x;
        origin_y_ = response.origin_m.y;
        origin_z_ = response.origin_m.z;

        voxel_size_m_ = response.voxel_size_m;

        size_x_ = array.layout.dim[0].size;
        size_y_ = array.layout.dim[1].size;
        size_z_ = array.layout.dim[2].size;

        stride_y_ = array.layout.dim[1].stride;
        stride_z_ = array.layout.dim[2].stride;

        data_ = array.data;

        const std::size_t expected_size = size_x_ * size_y_ * size_z_;

        if (voxel_size_m_ <= 0.0 ||
            size_x_ == 0 ||
            size_y_ == 0 ||
            size_z_ == 0 ||
            data_.size() != expected_size) {

            data_.clear();
            return false;
        }

        return true;
    }

    bool EsdfGrid::valid() const {
        return voxel_size_m_ > 0.0 && !data_.empty();
    }

    float EsdfGrid::distanceAt(const double x, const double y, const double z) const {
        if (!valid()) return std::numeric_limits<float>::quiet_NaN();

        // NVIDIA defines origin_m as the minimal corner of the minimal voxel.
        const int ix = static_cast<int>(std::floor((x - origin_x_) / voxel_size_m_));
        const int iy = static_cast<int>(std::floor((y - origin_y_) / voxel_size_m_));
        const int iz = static_cast<int>(std::floor((z - origin_z_) / voxel_size_m_));

        // Unknown / outside the requested ESDF is invalid.
        if (ix < 0 ||
            iy < 0 ||
            iz < 0 ||
            ix >= static_cast<int>(size_x_) ||
            iy >= static_cast<int>(size_y_) ||
            iz >= static_cast<int>(size_z_)) {

            return std::numeric_limits<float>::quiet_NaN();
        }

        // NVIDIA release-3.2 layout:
        //
        // index = x * dim[1].stride +
        //         y * dim[2].stride +
        //         z
        const std::size_t index = static_cast<std::size_t>(ix) * stride_y_ +
                                static_cast<std::size_t>(iy) * stride_z_ +
                                static_cast<std::size_t>(iz);

        if (index >= data_.size()) return std::numeric_limits<float>::quiet_NaN();

        return data_[index];
    }

    bool EsdfGrid::isStateValid(const double x,
                                const double y,
                                const double z,
                                const double clearance_m) const {

        const float distance_m = distanceAt(x, y, z);

        if (!std::isfinite(distance_m)) return false;

        // nvblox ESDF is signed:
        //   positive = outside obstacle
        //   negative = inside obstacle
        //
        // Unknown values such as the observed -1000 sentinel also fail this
        // positive-clearance test.
        return distance_m >= clearance_m;
    }

}  // namespace planning

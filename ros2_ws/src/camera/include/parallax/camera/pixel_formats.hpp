#pragma once

#include <cstdint>
#include <linux/videodev2.h>

#ifndef V4L2_PIX_FMT_BA10
#define V4L2_PIX_FMT_BA10 v4l2_fourcc('B', 'A', '1', '0')
#endif

namespace parallax::camera {

enum class BayerPattern : std::uint8_t {
    RGGB = 0,
    GRBG,
    GBRG,
    BGGR
};

}
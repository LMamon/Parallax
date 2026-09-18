#pragma once

namespace parallax::stereo {

    // Samples outside the useful metric envelope are treated as unknown geometry.
    inline constexpr float MinUsefulDepthM = 0.8F;
    inline constexpr float MaxUsefulDepthM = 8.0F;

}

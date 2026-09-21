#pragma once

namespace parallax::stereo {

    // Samples outside the useful metric envelope are treated as unknown geometry.
    inline constexpr float MinUsefulDepthM = 0.7F;
    inline constexpr float MaxUsefulDepthM = 9.0F;

}

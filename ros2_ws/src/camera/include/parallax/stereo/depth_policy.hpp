#pragma once


namespace parallax::stereo {

    // Samples outside the useful metric envelope are treated as unknown geometry.
    inline constexpr float MinUsefulDepthM = 0.5F;
    inline constexpr float MaxUsefulDepthM = 9.0F;

    // VPI confidence is U16 on a 0..65535 scale, with higher values indicating
    // a more reliable disparity. Start at 50% and tune against the physical rig.

}

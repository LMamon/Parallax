#pragma once

#include <parallax/application/command.hpp>

#include <string_view>

namespace parallax::application {

    // Parse the human-readable application command grammar.
    //
    // Supported commands:
    //   marker depth
    //   detect <target>
    //   track <target>
    //   stop tracking
    //
    // Parsing is deterministic and side-effect free. Invalid input must never
    // mutate application state or dependency-graph demand.
    [[nodiscard]] CommandParseResult parse_command(std::string_view input);
}
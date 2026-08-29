#pragma once

#include <parallax/application/command.hpp>

#include <string_view>

namespace parallax::application {

    /**
     * Decode the structured JSON command representation used by the
     * Foxglove control surface into the transport-independent Command type.
     */
    [[nodiscard]] CommandParseResult parse_foxglove_command(
        std::string_view message
    );

}
#pragma once

#include <cstdint>
#include <string>

namespace parallax::application {

    // Commands are application-level control inputs.
    //
    // A command describes a requested state transition. It does not identify
    // producers, execution policies, streams, or other runtime implementation
    // details. RequestController is responsible for translating commands into
    // application-owned state and dependency-graph demand.
    enum class CommandVerb : std::uint8_t {
        MarkerDepth, Detect, Track, StopTracking, Segment
    };

    // Commands may represent either a bounded request or persistent application
    // intent. Persistent intent remains active until another command changes it.
    enum class CommandBehavior : std::uint8_t {
        OneShot, Persistent
    };

    enum class DepthRequest : std::uint8_t {
        Unspecified, No, Yes
    };

    struct Command {
        CommandVerb verb;
        CommandBehavior behavior;
        std::string target;
        DepthRequest depth = DepthRequest::Unspecified;
    };

    enum class CommandParseError : std::uint8_t {
        None, Empty, UnknownCommand, MissingTarget, UnexpectedArgument
    };

    struct CommandParseResult {
        Command command{};
        CommandParseError error = CommandParseError::None;
        std::string message;

        [[nodiscard]] bool ok() const noexcept {
            return error == CommandParseError::None;
        }
    };
}
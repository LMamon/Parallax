#include <parallax/application/foxglove_command.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <string>

namespace parallax::application {

    CommandParseResult parse_foxglove_command(std::string_view message) {
        const auto json = nlohmann::json::parse(message.begin(), message.end(), nullptr, false);

        if (json.is_discarded() || !json.is_object()) {
            return {{}, CommandParseError::UnknownCommand, "command payload must be a JSON object"};
        }

        const auto command_it = json.find("command");
        const auto target_it = json.find("target");

        if (command_it == json.end() || !command_it->is_string()) {
            return {{}, CommandParseError::UnknownCommand, "command requires a string 'command' field"};
        }

        if (target_it == json.end() || !target_it->is_string()) {
            return {{}, CommandParseError::UnexpectedArgument, "command requires a string 'target' field"};
        }

        const std::string verb = command_it->get<std::string>();
        const std::string target = target_it->get<std::string>();

        if (verb == "navigation_goal") {
            if (!target.empty()) {
                return {{}, CommandParseError::UnexpectedArgument,
                        "navigation_goal does not accept a target"};
            }

            const auto x_it = json.find("x");
            const auto y_it = json.find("y");
            const auto z_it = json.find("z");

            if (x_it == json.end() || y_it == json.end() || z_it == json.end() ||
                !x_it->is_number() || !y_it->is_number() || !z_it->is_number()) {
                return {{}, CommandParseError::UnexpectedArgument,
                        "navigation_goal requires numeric x, y, z"};
            }

            const float x = x_it->get<float>();
            const float y = y_it->get<float>();
            const float z = z_it->get<float>();

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                return {{}, CommandParseError::UnexpectedArgument,
                        "navigation_goal coordinates must be finite"};
            }

            Command command{
                CommandVerb::NavigationGoal,
                CommandBehavior::Persistent,
                {}};
            command.position_m = {x, y, z};

            return {std::move(command), CommandParseError::None, {}};
        }

        if (verb == "marker_depth") {
            if (!target.empty()) {
                return {{}, CommandParseError::UnexpectedArgument, "marker_depth does not accept a target"};
            }

            return {Command{CommandVerb::MarkerDepth, CommandBehavior::Persistent, {}},
                            CommandParseError::None, {}};
        }

        if (verb == "detect") {
            if (target.empty()) {
                return {{}, CommandParseError::MissingTarget, "detect requires a target"};
            }

            return {Command{CommandVerb::Detect, CommandBehavior::OneShot, target},
                            CommandParseError::None, {}};
        }

        if (verb == "segment") {
            if (target.empty()) {
                return {{}, CommandParseError::MissingTarget, "segment requires a target"};
            }

            return {Command{CommandVerb::Segment, CommandBehavior::OneShot, target}, 
                            CommandParseError::None, {}};
        }

        if (verb == "track") {
            if (target.empty()) {
                return {{}, CommandParseError::MissingTarget, "track requires a target"};
            }

            return {Command{CommandVerb::Track, CommandBehavior::Persistent, target}, 
                            CommandParseError::None, {}};
        }

        if (verb == "stop_tracking") {
            if (!target.empty()) {
                return {{}, CommandParseError::UnexpectedArgument, "stop_tracking does not accept a target"};
            }

            return {Command{CommandVerb::StopTracking, CommandBehavior::OneShot, {}},
                            CommandParseError::None,{}};
        }

        return {{}, CommandParseError::UnknownCommand, "unknown command"};
    }
}
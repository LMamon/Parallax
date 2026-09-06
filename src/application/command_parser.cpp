#include <parallax/application/command_parser.hpp>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace parallax::application {
    namespace {
        std::string normalize(std::string_view input) {
            std::string result(input);

            std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                }
            );

            const auto first = result.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) {
                return {};
            }

            const auto last = result.find_last_not_of(" \t\r\n");
            return result.substr(first, last - first + 1);
        }

        std::vector<std::string> tokenize(const std::string& input) {
            std::istringstream stream(input);
            std::vector<std::string> tokens;
            std::string token;

            while (stream >> token) {
                tokens.push_back(std::move(token));
            }

            return tokens;
        }

        CommandParseResult failure(CommandParseError error, std::string message) {
            CommandParseResult result;
            result.error = error;
            result.message = std::move(message);
            return result;
        }

        bool parse_depth_option(const std::string& token, DepthRequest& depth) {
            constexpr std::string_view prefix = "depth=";
            if (token.rfind(prefix, 0) != 0) return false;

            const auto value = token.substr(prefix.size());

            if (value == "yes") {
                depth = DepthRequest::Yes;
                return true;
            }

            if (value == "no") {
                depth = DepthRequest::No;
                return true;
            }

            return false;
        }
    }

    CommandParseResult parse_command(std::string_view input) {
        const std::string normalized = normalize(input);

        if (normalized.empty()) {
            return failure(CommandParseError::Empty, "command is empty");
        }

        const auto tokens = tokenize(normalized);

        if (tokens[0] == "marker") {
            if (tokens.size() == 1) {
                return failure(CommandParseError::MissingTarget, "expected 'marker depth'");
            }

            if (tokens.size() != 2 || tokens[1] != "depth") {
                return failure(CommandParseError::UnexpectedArgument, "expected 'marker depth'");
            }

            return {Command{CommandVerb::MarkerDepth,CommandBehavior::OneShot, {}}, CommandParseError::None, {}};
        }

        if (tokens[0] == "detect") {
            if (tokens.size() < 2) {
                return failure(CommandParseError::MissingTarget, "detect requires a target");
            }

            if (tokens.size() > 3) {
                return failure(CommandParseError::UnexpectedArgument, "detect accepts one target and optional depth=yes|no");
            }
            DepthRequest depth = DepthRequest::Unspecified;

            if (tokens.size() == 3 && !parse_depth_option(tokens[2], depth)) {
                return failure(CommandParseError::UnexpectedArgument, "detect accepts optional depth=yes|no");
            }

            return {Command{CommandVerb::Detect, CommandBehavior::OneShot, tokens[1], depth}, CommandParseError::None, {}
            };
        }

        if (tokens[0] == "segment") {
            if (tokens.size() < 2) {
                return failure(CommandParseError::MissingTarget, "segment requires a target");
            }

            if (tokens.size() > 2) {
                return failure(CommandParseError::UnexpectedArgument, "segment accepts exactly one target");
            }

            return {Command{CommandVerb::Segment, CommandBehavior::OneShot, tokens[1]}, CommandParseError::None, {}};
        }

        if (tokens[0] == "track") {
            if (tokens.size() < 2) {
                return failure(CommandParseError::MissingTarget, "track requires a target");
            }

            if (tokens.size() > 2) {
                return failure(CommandParseError::UnexpectedArgument,"track accepts exactly one target");
            }

            return {Command{CommandVerb::Track, CommandBehavior::Persistent, tokens[1]}, CommandParseError::None, {}};
        }

        if (tokens[0] == "stop") {
            if (tokens.size() != 2 || tokens[1] != "tracking") {
                return failure(CommandParseError::UnexpectedArgument, "expected 'stop tracking'");
            }

            return {Command{CommandVerb::StopTracking, CommandBehavior::OneShot, {}}, CommandParseError::None, {}};
        }

        return failure(CommandParseError::UnknownCommand, "unknown command");
    }
}
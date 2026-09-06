#include <parallax/application/command_parser.hpp>

#include <gtest/gtest.h>

namespace {

    namespace application = parallax::application;

    TEST(CommandParserTest, DetectDefaultsDepthToUnspecified) {
        const auto result = application::parse_command("detect cup");

        ASSERT_TRUE(result.ok());
        EXPECT_EQ(result.command.verb, application::CommandVerb::Detect);
        EXPECT_EQ(result.command.target, "cup");
        EXPECT_EQ(result.command.depth, application::DepthRequest::Unspecified);
    }

    TEST(CommandParserTest, DetectAcceptsDepthYes) {
        const auto result = application::parse_command("detect cup depth=yes");

        ASSERT_TRUE(result.ok());
        EXPECT_EQ(result.command.target, "cup");
        EXPECT_EQ(result.command.depth, application::DepthRequest::Yes);
    }

    TEST(CommandParserTest, DetectAcceptsDepthNo) {
        const auto result = application::parse_command("detect cup depth=no");

        ASSERT_TRUE(result.ok());
        EXPECT_EQ(result.command.target, "cup");
        EXPECT_EQ(result.command.depth, application::DepthRequest::No);
    }

    TEST(CommandParserTest, DetectRejectsInvalidDepthValue) {
        const auto result = application::parse_command("detect cup depth=maybe");

        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.error, application::CommandParseError::UnexpectedArgument);
    }

    TEST(CommandParserTest, DetectRejectsUnknownModifier) {
        const auto result = application::parse_command("detect cup foo=yes");

        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.error, application::CommandParseError::UnexpectedArgument);
    }
}
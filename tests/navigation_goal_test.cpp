#include <parallax/application/foxglove_command.hpp>
#include <parallax/application/navigation_goal.hpp>
#include <parallax/application/request_controller.hpp>
#include <parallax/core/dependency_resolver.hpp>
#include <parallax/core/graph.hpp>

#include <gtest/gtest.h>

#include <limits>

namespace {

using parallax::application::Command;
using parallax::application::CommandBehavior;
using parallax::application::CommandVerb;
using parallax::application::NavigationGoal;
using parallax::application::RequestController;
using parallax::application::RequestStatus;
using parallax::core::DependencyResolver;
using parallax::core::Graph;

TEST(NavigationGoalTest, PayloadRequiresFiniteVersionedPosition) {
    NavigationGoal goal;
    EXPECT_FALSE(goal.valid());

    goal.position_m = {1.0F, -2.0F, 0.5F};
    goal.revision = 1;
    EXPECT_TRUE(goal.valid());

    goal.position_m[1] =
        std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(goal.valid());
}

TEST(NavigationGoalTest, FoxgloveCommandParsesWorldPosition) {
    const auto result =
        parallax::application::parse_foxglove_command(
            R"({"command":"navigation_goal","target":"","x":1.25,"y":-2.0,"z":0.75})");

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.command.verb, CommandVerb::NavigationGoal);
    EXPECT_FLOAT_EQ(result.command.position_m[0], 1.25F);
    EXPECT_FLOAT_EQ(result.command.position_m[1], -2.0F);
    EXPECT_FLOAT_EQ(result.command.position_m[2], 0.75F);
}

TEST(NavigationGoalTest, FoxgloveCommandRejectsMissingCoordinate) {
    const auto result =
        parallax::application::parse_foxglove_command(
            R"({"command":"navigation_goal","target":"","x":1.0,"y":2.0})");

    EXPECT_FALSE(result.ok());
}

TEST(NavigationGoalTest, ControllerReplacesGoalAndAdvancesRevision) {
    Graph graph;
    DependencyResolver resolver(graph);
    RequestController controller(resolver);

    Command first{
        CommandVerb::NavigationGoal,
        CommandBehavior::Persistent,
        {}};
    first.position_m = {1.0F, 2.0F, 3.0F};

    ASSERT_EQ(controller.apply(first).status,
              RequestStatus::Applied);

    const auto first_state = controller.state();
    EXPECT_TRUE(first_state.navigation_goal_requested);
    EXPECT_EQ(first_state.navigation_goal_revision, 1U);

    Command second{
        CommandVerb::NavigationGoal,
        CommandBehavior::Persistent,
        {}};
    second.position_m = {-1.0F, 4.0F, 0.25F};

    ASSERT_EQ(controller.apply(second).status,
              RequestStatus::Applied);

    const auto second_state = controller.state();
    EXPECT_EQ(second_state.navigation_goal_revision, 2U);
    EXPECT_FLOAT_EQ(second_state.navigation_goal_m[0], -1.0F);
    EXPECT_FLOAT_EQ(second_state.navigation_goal_m[1], 4.0F);
    EXPECT_FLOAT_EQ(second_state.navigation_goal_m[2], 0.25F);
}

}  // namespace

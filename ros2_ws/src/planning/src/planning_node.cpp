#include <algorithm>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <nvblox_msgs/srv/esdf_and_gradients.hpp>

#include <ompl/base/ScopedState.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>

#include "planning/esdf_grid.hpp"
#include "planning/esdf_motion_validator.hpp"

namespace ob = ompl::base;
namespace og = ompl::geometric;

class PlannerNode : public rclcpp::Node {
    public:
        PlannerNode() : Node("planner"), tf_buffer_(get_clock()),
                                         tf_listener_(tf_buffer_),
                                         esdf_grid_(std::make_shared<planning::EsdfGrid>()) {

            clearance_m_ = declare_parameter<double>("clearance_m", 0.10);
            aabb_padding_m_ = declare_parameter<double>("aabb_padding_m", 1.0);
            solve_time_s_ = declare_parameter<double>("solve_time_s", 1.5);
            motion_check_resolution_m_ = declare_parameter<double>("motion_check_resolution_m", 0.05);

            global_frame_ = declare_parameter<std::string>("global_frame", "map");
            base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
            goal_topic_ = declare_parameter<std::string>("goal_topic", "/move_base_simple/goal");
            path_topic_ = declare_parameter<std::string>("path_topic", "/planned_path_3d");
            esdf_service_ = declare_parameter<std::string>("esdf_service", "/nvblox_node/get_esdf_and_gradient");

            update_esdf_ = declare_parameter<bool>("update_esdf", true);
            visualize_esdf_ = declare_parameter<bool>("visualize_esdf", false);

            if (clearance_m_ < 0.0 ||
                aabb_padding_m_ <= 0.0 ||
                solve_time_s_ <= 0.0 ||
                motion_check_resolution_m_ <= 0.0) {

                throw std::runtime_error("Planner numeric parameters must be positive "
                    "(clearance_m may be zero)");
            }

            goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
                                            goal_topic_,
                                            10,
                                            std::bind(
                                                &PlannerNode::goalCallback,
                                                this,
                                                std::placeholders::_1));

            path_pub_ = create_publisher<nav_msgs::msg::Path>(path_topic_, 10);
            esdf_client_ = create_client<nvblox_msgs::srv::EsdfAndGradients>(esdf_service_);

            RCLCPP_INFO(get_logger(),
                        "Planner ready: clearance=%.3f m, AABB padding=%.2f m, "
                        "motion check=%.3f m",
                        clearance_m_,
                        aabb_padding_m_,
                        motion_check_resolution_m_);
        }

    private:
        void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            geometry_msgs::msg::PoseStamped goal_map;

            try {
                goal_map = tf_buffer_.transform(*msg, global_frame_, tf2::durationFromSec(0.1));
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN(get_logger(),
                            "Cannot transform goal from '%s' to '%s': %s",
                            msg->header.frame_id.c_str(),
                            global_frame_.c_str(),
                            ex.what());
                return;
            }

            geometry_msgs::msg::TransformStamped transform;

            try {
                transform = tf_buffer_.lookupTransform(global_frame_, base_frame_, tf2::TimePointZero);
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN(get_logger(),
                            "Cannot plan: %s->%s unavailable: %s",
                            global_frame_.c_str(),
                            base_frame_.c_str(),
                            ex.what());
                return;
            }

            start_x_ = transform.transform.translation.x;
            start_y_ = transform.transform.translation.y;
            start_z_ = transform.transform.translation.z;

            goal_x_ = goal_map.pose.position.x;
            goal_y_ = goal_map.pose.position.y;
            goal_z_ = goal_map.pose.position.z;

            RCLCPP_INFO(get_logger(),
                        "Planning [%.2f %.2f %.2f] -> [%.2f %.2f %.2f]",
                        start_x_,
                        start_y_,
                        start_z_,
                        goal_x_,
                        goal_y_,
                        goal_z_);

            requestEsdf();
        }

        void requestEsdf() {
            if (!esdf_client_->service_is_ready()) {
                RCLCPP_WARN(get_logger(), "nvblox ESDF service is not ready");
                return;
            }

            auto request = std::make_shared<nvblox_msgs::srv::EsdfAndGradients::Request>();

            request->update_esdf = update_esdf_;
            request->visualize_esdf = visualize_esdf_;
            request->use_aabb = true;
            request->frame_id = global_frame_;

            const double min_x = std::min(start_x_, goal_x_) - aabb_padding_m_;
            const double min_y = std::min(start_y_, goal_y_) - aabb_padding_m_;
            const double min_z = std::min(start_z_, goal_z_) - aabb_padding_m_;

            const double max_x = std::max(start_x_, goal_x_) + aabb_padding_m_;
            const double max_y = std::max(start_y_, goal_y_) + aabb_padding_m_;
            const double max_z = std::max(start_z_, goal_z_) + aabb_padding_m_;

            request->aabb_min_m.x = min_x;
            request->aabb_min_m.y = min_y;
            request->aabb_min_m.z = min_z;

            request->aabb_size_m.x = max_x - min_x;
            request->aabb_size_m.y = max_y - min_y;
            request->aabb_size_m.z = max_z - min_z;

            planning_min_x_ = min_x;
            planning_min_y_ = min_y;
            planning_min_z_ = min_z;

            planning_max_x_ = max_x;
            planning_max_y_ = max_y;
            planning_max_z_ = max_z;

            RCLCPP_INFO(get_logger(),
                        "Requesting ESDF AABB: min=[%.2f %.2f %.2f], "
                        "size=[%.2f %.2f %.2f]",
                        min_x,
                        min_y,
                        min_z,
                        request->aabb_size_m.x,
                        request->aabb_size_m.y,
                        request->aabb_size_m.z);

            esdf_client_->async_send_request(request,
                                            std::bind(
                                                &PlannerNode::esdfCallback,
                                                this,
                                                std::placeholders::_1));
        }

        void esdfCallback(rclcpp::Client<nvblox_msgs::srv::EsdfAndGradients>::SharedFuture future) {
            nvblox_msgs::srv::EsdfAndGradients::Response::SharedPtr response;

            try {
                response = future.get();
            } catch (const std::exception &ex) {
                RCLCPP_ERROR(get_logger(), "ESDF service request failed: %s", ex.what());
                return;
            }

            if (!response->success) {
                RCLCPP_WARN(get_logger(), "nvblox failed to create requested ESDF grid");
                return;
            }

            if (!esdf_grid_->update(*response)) {
                RCLCPP_ERROR(get_logger(), "Received malformed or unusable ESDF grid");
                return;
            }

            const auto &dims = response->esdf_and_gradients.layout.dim;

            RCLCPP_INFO(get_logger(),
                        "ESDF received: %ux%ux%u voxels @ %.3f m",
                        dims[0].size,
                        dims[1].size,
                        dims[2].size,
                        response->voxel_size_m);

            plan();
        }

        bool isStateValid(const ob::State *state) const {
            const auto *xyz = state->as<ob::RealVectorStateSpace::StateType>();

            return esdf_grid_->isStateValid(xyz->values[0],
                                            xyz->values[1],
                                            xyz->values[2],
                                            clearance_m_);
        }

        void plan() {
            auto space = std::make_shared<ob::RealVectorStateSpace>(3);

            ob::RealVectorBounds bounds(3);

            bounds.setLow(0, planning_min_x_);
            bounds.setHigh(0, planning_max_x_);

            bounds.setLow(1, planning_min_y_);
            bounds.setHigh(1, planning_max_y_);

            bounds.setLow(2, planning_min_z_);
            bounds.setHigh(2, planning_max_z_);

            space->setBounds(bounds);

            og::SimpleSetup setup(space);

            setup.setStateValidityChecker(
                [this](const ob::State *state) {
                    return isStateValid(state);
                });

            auto si = setup.getSpaceInformation();

            // RRTConnect may propose long edges. Check those edges against the
            // cached ESDF at a fixed physical interval instead of relying on
            // endpoint validity alone.
            si->setMotionValidator(
                std::make_shared<planning::EsdfMotionValidator>(
                    si,
                    esdf_grid_,
                    clearance_m_,
                    motion_check_resolution_m_));

            setup.setPlanner(std::make_shared<og::RRTConnect>(si));

            ob::ScopedState<ob::RealVectorStateSpace> start(space);
            start[0] = start_x_;
            start[1] = start_y_;
            start[2] = start_z_;

            ob::ScopedState<ob::RealVectorStateSpace> goal(space);
            goal[0] = goal_x_;
            goal[1] = goal_y_;
            goal[2] = goal_z_;

            setup.setStartAndGoalStates(start, goal);

            if (!si->isValid(start.get())) {
                const float d = esdf_grid_->distanceAt(
                                start_x_,
                                start_y_,
                                start_z_);

                RCLCPP_WARN(get_logger(),
                            "Cannot plan: current %s position is not ESDF-valid "
                            "(distance=%.3f m, clearance=%.3f m)",
                            base_frame_.c_str(),
                            d,
                            clearance_m_);
                return;
            }

            if (!si->isValid(goal.get())) {
                const float d = esdf_grid_->distanceAt(goal_x_, goal_y_, goal_z_);

                RCLCPP_WARN(get_logger(),
                            "Cannot plan: requested goal is not ESDF-valid "
                            "(distance=%.3f m, clearance=%.3f m)",
                            d,
                            clearance_m_);
                return;
            }

            const ob::PlannerStatus solved = setup.solve(solve_time_s_);

            if (!solved) {
                RCLCPP_WARN(get_logger(), "RRTConnect found no path");
                return;
            }

            const auto &solution = setup.getSolutionPath();

            RCLCPP_INFO(get_logger(),
                        "Path found: %zu states, %.2f m",
                        solution.getStateCount(),
                        solution.length());

            publishPath(solution);
        }

        void publishPath(const og::PathGeometric &path) {
            nav_msgs::msg::Path msg;

            msg.header.stamp = now();
            msg.header.frame_id = global_frame_;

            for (std::size_t i = 0; i < path.getStateCount(); ++i) {

                const auto *state = path.getState(i)->as<ob::RealVectorStateSpace::StateType>();

                geometry_msgs::msg::PoseStamped pose;
                pose.header = msg.header;

                pose.pose.position.x = state->values[0];
                pose.pose.position.y = state->values[1];
                pose.pose.position.z = state->values[2];

                pose.pose.orientation.w = 1.0;

                msg.poses.push_back(pose);
            }

            path_pub_->publish(msg);

            RCLCPP_INFO(get_logger(), "Published %s with %zu poses", path_topic_.c_str(), msg.poses.size());
        }

        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
        rclcpp::Client<nvblox_msgs::srv::EsdfAndGradients>::SharedPtr esdf_client_;

        tf2_ros::Buffer tf_buffer_;
        tf2_ros::TransformListener tf_listener_;

        std::shared_ptr<planning::EsdfGrid> esdf_grid_;

        double start_x_{0.0};
        double start_y_{0.0};
        double start_z_{0.0};

        double goal_x_{0.0};
        double goal_y_{0.0};
        double goal_z_{0.0};

        double planning_min_x_{0.0};
        double planning_min_y_{0.0};
        double planning_min_z_{0.0};

        double planning_max_x_{0.0};
        double planning_max_y_{0.0};
        double planning_max_z_{0.0};

        double clearance_m_{0.10};
        double aabb_padding_m_{1.0};
        double solve_time_s_{1.5};
        double motion_check_resolution_m_{0.05};

        std::string global_frame_{"map"};
        std::string base_frame_{"base_link"};
        std::string goal_topic_{"/move_base_simple/goal"};
        std::string path_topic_{"/planned_path_3d"};
        std::string esdf_service_{"/nvblox_node/get_esdf_and_gradient"};

        bool update_esdf_{true};
        bool visualize_esdf_{false};
    };

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PlannerNode>());
    rclcpp::shutdown();
    return 0;
}

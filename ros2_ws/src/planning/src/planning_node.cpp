#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point_stamped.hpp>
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


namespace ob = ompl::base;
namespace og = ompl::geometric;

// The important architectural point is that OMPL does NOT call nvblox for
// every state it tests.
//
// We request one bounded ESDF grid from nvblox and then OMPL performs all of
// its state-validity checks against that local CPU-side grid.
//
class PlannerNode : public rclcpp::Node {
    public:
        PlannerNode() : Node("planner"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_) {
            // clearance_m:
            // Minimum ESDF distance that a state must have from an obstacle.
            //
            // aabb_padding_m:
            // Extra map volume requested around the start->goal region.
            //
            // solve_time_s:
            // Maximum time RRTConnect gets for a planning attempt.

            clearance_m_ = declare_parameter<double>("clearance_m", 0.25);
            aabb_padding_m_ = declare_parameter<double>("aabb_padding_m", 1.0);
            solve_time_s_ = declare_parameter<double>("solve_time_s", 1.5);


            goal_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
                                            "/clicked_point",
                                            10,
                                            std::bind(
                                                &PlannerNode::goalCallback,
                                                this,
                                                std::placeholders::_1));

            // Add /planned_path_3d to the Foxglove whitelist.
            path_pub_ = create_publisher<nav_msgs::msg::Path>("/planned_path_3d", 10);

 
            // nvblox already runs in your spatial container.
            // We are only consuming its map through the supported service.
            esdf_client_ = create_client<nvblox_msgs::srv::EsdfAndGradients>("/nvblox_node/get_esdf_and_gradient");

            RCLCPP_INFO(get_logger(), "Planner ready: clearance=%.2f m, AABB padding=%.2f m", clearance_m_, aabb_padding_m_);
        }


    private:
        void goalCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
            geometry_msgs::msg::PointStamped goal_map;
            try {
                goal_map = tf_buffer_.transform(*msg, // Point received from Foxglove
                                                "map", // Frame you want the point expressed in
                                                tf2::durationFromSec(0.1)); // Maximum time to wait for TF
            
            } catch (const tf2::TransformException &ex) {
                // Planning cannot continue unless the goal and ESDF use the same
                // coordinate frame. Reject this goal rather than using bad geometry.
                RCLCPP_WARN(get_logger(), "Cannot transform goal from '%s' to 'map': %s",
                            msg->header.frame_id.c_str(), ex.what());
                return;
            }

            // TF gives planning the current base_link position directly in the
            // same coordinate frame as the nvblox map.
            geometry_msgs::msg::TransformStamped transform;

            try {
                transform = tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
            }
            catch (const tf2::TransformException &ex) {
                RCLCPP_WARN(get_logger(), "Cannot plan: map->base_link unavailable: %s", ex.what());

                return;
            }

            start_x_ = transform.transform.translation.x;
            start_y_ = transform.transform.translation.y;
            start_z_ = transform.transform.translation.z;

            goal_x_ = goal_map.point.x;
            goal_y_ = goal_map.point.y;
            goal_z_ = goal_map.point.z;

            RCLCPP_INFO(get_logger(), "Goal transformed from '%s' to 'map'", msg->header.frame_id.c_str());
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

            // make sure ESDF is current before returning it.
            request->update_esdf = true;

            // don't need nvblox to publish additional visualization just
            // because planning requested a region.
            request->visualize_esdf = false;

            // only want the part of the world relevant to this plan.
            request->use_aabb = true;
            request->frame_id = "map";


            // Build an AABB containing start and goal
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

            // Save these bounds because OMPL should search exactly the region
            // represented by the ESDF requested.
            planning_min_x_ = min_x;
            planning_min_y_ = min_y;
            planning_min_z_ = min_z;

            planning_max_x_ = max_x;
            planning_max_y_ = max_y;
            planning_max_z_ = max_z;

            RCLCPP_INFO(get_logger(),
                        "Requesting ESDF AABB: min=[%.2f %.2f %.2f], size=[%.2f %.2f %.2f]",
                        min_x,
                        min_y,
                        min_z,
                        request->aabb_size_m.x,
                        request->aabb_size_m.y,
                        request->aabb_size_m.z);

            // Asynchronous service call
            // Do NOT block the ROS executor waiting for nvblox.
            esdf_client_->async_send_request(request,
                                            std::bind(
                                                &PlannerNode::esdfCallback,
                                                this,
                                                std::placeholders::_1));
        }


        // Receive and cache the ESDF
        void esdfCallback(rclcpp::Client<nvblox_msgs::srv::EsdfAndGradients>::SharedFuture future) {
            const auto response = future.get();

            if (!response->success) {
                RCLCPP_WARN(get_logger(), "nvblox failed to create requested ESDF grid");

                return;
            }

            const auto &array = response->esdf_and_gradients;

            // NVIDIA release-3.2 constructs exactly three dimensions:
            //
            // dim[0] = x
            // dim[1] = y
            // dim[2] = z
            if (array.layout.dim.size() != 3) {
                RCLCPP_ERROR(get_logger(), "Unexpected ESDF dimension count: %zu", array.layout.dim.size());

                return;
            }


            if (array.layout.dim[0].label != "x" ||
                array.layout.dim[1].label != "y" ||
                array.layout.dim[2].label != "z") {

                RCLCPP_ERROR(get_logger(), "Unexpected ESDF dimension labels");

                return;
            }

            // Cache the grid metadata
            esdf_origin_x_ = response->origin_m.x;
            esdf_origin_y_ = response->origin_m.y;
            esdf_origin_z_ = response->origin_m.z;

            esdf_voxel_size_ = response->voxel_size_m;

            esdf_size_x_ = array.layout.dim[0].size;
            esdf_size_y_ = array.layout.dim[1].size;
            esdf_size_z_ = array.layout.dim[2].size;

            esdf_stride_y_ = array.layout.dim[1].stride;
            esdf_stride_z_ = array.layout.dim[2].stride;

            // This is the actual dense CPU-side ESDF cache used by OMPL.
            esdf_data_ = array.data;

            const std::size_t expected_size = esdf_size_x_ * esdf_size_y_ * esdf_size_z_;

            if (esdf_data_.size() != expected_size) {
                RCLCPP_ERROR(get_logger(), "ESDF size mismatch: got %zu floats, expected %zu", esdf_data_.size(), expected_size);
                esdf_data_.clear();

                return;
            }


            RCLCPP_INFO(get_logger(),
                        "ESDF received: %zux%zux%zu voxels @ %.3f m",
                        esdf_size_x_,
                        esdf_size_y_,
                        esdf_size_z_,
                        esdf_voxel_size_);

            plan();
        }

        // Query local ESDF cache
        bool isStateValid(const ob::State *state) const {
            const auto *xyz = state->as<ob::RealVectorStateSpace::StateType>();

            const double x = xyz->values[0];
            const double y = xyz->values[1];
            const double z = xyz->values[2];

            // World coordinate -> voxel coordinate
            // NVIDIA defines origin_m as the MINIMAL CORNER of the minimal voxel.
            //
            //     voxel_x = floor((world_x - origin_x) / voxel_size)
            const int ix = static_cast<int>(std::floor((x - esdf_origin_x_) / esdf_voxel_size_));
            const int iy = static_cast<int>(std::floor((y - esdf_origin_y_) / esdf_voxel_size_));
            const int iz = static_cast<int>(std::floor((z - esdf_origin_z_) / esdf_voxel_size_));


            // Outside our requested ESDF region = not valid.
            //
            // This keeps OMPL conservative rather than allowing it to escape the
            // known planning volume.
            if (ix < 0 ||
                iy < 0 ||
                iz < 0 ||
                ix >= static_cast<int>(esdf_size_x_) ||
                iy >= static_cast<int>(esdf_size_y_) ||
                iz >= static_cast<int>(esdf_size_z_)) {

                return false;
            }

            // NVIDIA release-3.2 indexing
            // Their own unit test uses:
            // 
            // index = x * dim[1].stride +
            //         y * dim[2].stride +
            //         z;

            const std::size_t index = static_cast<std::size_t>(ix) * esdf_stride_y_ +
                                      static_cast<std::size_t>(iy) * esdf_stride_z_ +
                                      static_cast<std::size_t>(iz);

            if (index >= esdf_data_.size()) return false;

            const float distance_m = esdf_data_[index];


            // TEMP diagnostic: show exactly what ESDF value we're testing.

            RCLCPP_INFO(get_logger(),
                        "ESDF sample: world=[%.3f %.3f %.3f] voxel=[%d %d %d] "
                        "index=%zu distance=%.3f m clearance=%.3f m",
                        x, y, z,
                        ix, iy, iz,
                        index,
                        distance_m,
                        clearance_m_);
                        
            // Collision / clearance test
            // https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/blob/release-3.2/nvblox_ros/src/lib/conversions/esdf_and_gradients_conversions.cu
            // NVIDIA's converter produces SIGNED distances:
            //
            //     positive = outside obstacle
            //     negative = inside obstacle
            //
            // Unobserved voxels use nvblox's configured default value. Treating
            // anything below positive clearance threshold as invalid also
            // makes unknown/unobserved space conservative.

            if (!std::isfinite(distance_m)) return false;

            return distance_m >= clearance_m_;
        }


        // Run OMPL
        void plan() {
            // XYZ state space
            // only need position for this planner.
            auto space = std::make_shared<ob::RealVectorStateSpace>(3);


            // OMPL is bounded to the same physical region represented by the
            // cached ESDF.
            ob::RealVectorBounds bounds(3);

            bounds.setLow(0, planning_min_x_);
            bounds.setHigh(0, planning_max_x_);

            bounds.setLow(1, planning_min_y_);
            bounds.setHigh(1, planning_max_y_);

            bounds.setLow(2, planning_min_z_);
            bounds.setHigh(2, planning_max_z_);

            space->setBounds(bounds);

            og::SimpleSetup setup(space);


            // Every RRTConnect sample comes through here.
            //
            // This is now a cheap local array lookup -- no ROS service call.
            //
            setup.setStateValidityChecker(
                [this](const ob::State *state) {
                    return isStateValid(state);
                });


            setup.setPlanner(std::make_shared<og::RRTConnect>(setup.getSpaceInformation()));

            // Start
            ob::ScopedState<ob::RealVectorStateSpace> start(space);
            start[0] = start_x_;
            start[1] = start_y_;
            start[2] = start_z_;

            // Goal
            ob::ScopedState<ob::RealVectorStateSpace> goal(space);
            goal[0] = goal_x_;
            goal[1] = goal_y_;
            goal[2] = goal_z_;

            setup.setStartAndGoalStates(start, goal);

            // Sanity check before searching
            if (!setup.getSpaceInformation()->isValid(start.get())) {
                RCLCPP_WARN(get_logger(), "Cannot plan: current base_link position is not ESDF-valid");

                return;
            }

            if (!setup.getSpaceInformation()->isValid(goal.get())) {
                RCLCPP_WARN(get_logger(), "Cannot plan: requested goal is not ESDF-valid");

                return;
            }

            // Solve
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

        // OMPL PathGeometric -> nav_msgs/Path
        void publishPath(const og::PathGeometric &path) {
            nav_msgs::msg::Path msg;

            msg.header.stamp = now();
            msg.header.frame_id = "map";

            for (std::size_t i = 0; i < path.getStateCount(); ++i) {
                const auto *state = path.getState(i)->as<ob::RealVectorStateSpace::StateType>();

                geometry_msgs::msg::PoseStamped pose;
                pose.header = msg.header;
                pose.pose.position.x = state->values[0];
                pose.pose.position.y = state->values[1];
                pose.pose.position.z = state->values[2];

                // Orientation is not part of this planning state space.
                // Identity quaternion keeps PoseStamped valid.
                pose.pose.orientation.w = 1.0;

                msg.poses.push_back(pose);
            }

            path_pub_->publish(msg);
            RCLCPP_INFO(get_logger(), "Published /planned_path_3d with %zu poses", msg.poses.size());
        }

        // ROS
        rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_sub_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
        rclcpp::Client<nvblox_msgs::srv::EsdfAndGradients>::SharedPtr esdf_client_;

        // TF
        tf2_ros::Buffer tf_buffer_;
        tf2_ros::TransformListener tf_listener_;

        double start_x_{0.0};
        double start_y_{0.0};
        double start_z_{0.0};

        double goal_x_{0.0};
        double goal_y_{0.0};
        double goal_z_{0.0};


        // Bounds used by both nvblox and OMPL.
        double planning_min_x_{0.0};
        double planning_min_y_{0.0};
        double planning_min_z_{0.0};

        double planning_max_x_{0.0};
        double planning_max_y_{0.0};
        double planning_max_z_{0.0};

        // Local ESDF cache
        double esdf_origin_x_{0.0};
        double esdf_origin_y_{0.0};
        double esdf_origin_z_{0.0};

        double esdf_voxel_size_{0.0};

        std::size_t esdf_size_x_{0};
        std::size_t esdf_size_y_{0};
        std::size_t esdf_size_z_{0};

        std::size_t esdf_stride_y_{0};
        std::size_t esdf_stride_z_{0};

        std::vector<float> esdf_data_;

        // Planner configuration
        double clearance_m_{0.25};
        double aabb_padding_m_{1.0};
        double solve_time_s_{1.5};
    };


int main(int argc, char **argv) {
        rclcpp::init(argc, argv);
        rclcpp::spin(std::make_shared<PlannerNode>());
        rclcpp::shutdown();
        return 0;
    }
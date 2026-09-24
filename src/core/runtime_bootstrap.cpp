#include <parallax/core/runtime.hpp>
#include <parallax/core/pipeline.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/pose/charuco_pose.hpp>
#include <parallax/core/history_configuration.hpp>
#include <parallax/core/runtime_metrics.hpp>
#include <parallax/mapping/mapping_metrics.hpp>
#include <parallax/application/foxglove_command.hpp>
#include <parallax/application/navigation_goal.hpp>
#include <parallax/camera/arducam_controls.hpp>

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string_view>
#include <limits>
#include <vector>
#include <iostream>

namespace parallax::core {
bool Runtime::initialize(const std::filesystem::path& camera_config_path,
                             const std::filesystem::path& isp_config_path,
                             const std::filesystem::path& sensor_extrinsics_path,
                             const std::filesystem::path& mapping_config_path,
                             const std::filesystem::path& calibration_directory,
                             const std::filesystem::path& nanoowl_engine_path) {

        if (initialized_) return true;

        if (!context_.initialize()) {
            std::cerr << "Runtime: failed to initialize execution context\n";
            shutdown();
            return false;
        }

        if (!config_.loadFromFile(camera_config_path)) {
            std::cerr << "Runtime: failed to load camera config\n";
            return false;
        }

        if (!isp_config_.loadFromFile(isp_config_path)) {
            std::cerr << "Runtime: failed to load ISP config\n";
            return false;
        }

        if (!sensor_extrinsics_.loadFromFile(sensor_extrinsics_path)) {
            std::cerr << "Runtime: failed to load extrinsics config\n";
            return false;
        }

        if (!mapping_config_.loadFromFile(mapping_config_path)) {
            std::cerr << "Runtime: failed to load mapping config\n";
            return false;
        }

        camera_ = std::make_unique<parallax::camera::StereoCamera>(config_);
        if (!camera_->initialize()) {
            std::cerr << "Runtime: failed to initialize camera\n";
            shutdown();
            return false;
        }

        lidar_ = std::make_unique<parallax::lidar::Rplidar>();
        if (!lidar_->initialize()) {
            std::cerr << "RPLIDAR: no device detected; continuing without LiDAR\n";
            lidar_.reset();
        }
        /**
         * ISP allocations, VPI stream, rectifier, matcher, depth storage, and pose
         * estimator. Runtime now takes over orchestration through graph producers.
         */
        if (!pipeline_.initialize(config_, isp_config_, calibration_directory)) {
            std::cerr << "Runtime: failed to initialize processing pipeline\n";
            shutdown();
            return false;
        }

        if (isp_config_.auto_exposure.enable || isp_config_.auto_white_balance.enable) {
            parallax::camera::ControlRange exposure_range{};
            parallax::camera::ControlRange gain_range{};

            if (isp_config_.auto_exposure.enable) {
                if (!camera_->getControlRange(parallax::camera::controls::Exposure, exposure_range) ||
                    !camera_->getControlRange(parallax::camera::controls::AnalogGain, gain_range)) {
                    std::cerr << "Runtime: failed to query exposure/gain control ranges\n";
                    shutdown();
                    return false;
                }
            } else {
                exposure_range = {config_.exposure, config_.exposure, 1, config_.exposure, true};
                gain_range = {config_.analogue_gain, config_.analogue_gain, 1, config_.analogue_gain, true};
            }
            auto_controller_ = std::make_unique<parallax::isp::AutoController>(
                isp_config_, exposure_range, gain_range, config_.exposure, config_.analogue_gain);
        }

        cuvslam_localizer_ = std::make_unique<parallax::localization::CuVslamLocalizer>();
        if (!cuvslam_localizer_->initialize(pipeline_.calibration(), sensor_extrinsics_)) {
            std::cerr << "Runtime: failed to initialize cuVSLAM\n";
            shutdown();
            return false;
        }

        nanoowl_ = std::make_unique<parallax::perception::NanoOwlBridge>();
        if (!nanoowl_->initialize(nanoowl_engine_path)) {
            std::cerr << "Runtime: failed to initialize NanoOWL\n";
            shutdown();
            return false;
        }

        stereo_roi_associator_ = std::make_unique<parallax::perception::StereoRoiAssociator>(pipeline_.calibration(), sensor_extrinsics_.left_camera.child_frame);
        if (!stereo_roi_associator_->initialize()) {
            std::cerr << "Runtime: failed to initialize stereo ROI associator\n";
            shutdown();
            return false;
        }

        lidar_detection_associator_ = std::make_unique<parallax::perception::LidarDetectionAssociator>();

        if (!lidar_detection_associator_->initialize(pipeline_.calibration(),
                                                      sensor_extrinsics_,
                                                      sensor_extrinsics_.left_camera.child_frame)) {
            std::cerr << "Runtime: failed to initialize LiDAR detection associator\n";
            shutdown();
            return false;
        }

        efficientvit_sam_ = std::make_unique<parallax::perception::EfficientVitSam>();

        /**
         * Construct the complete producer set before registering or finalizing the
         * graph. Graph stores non-owning Producer pointers, so Runtime owns every
         * producer for the lifetime of the graph.
         */
        camera_producer_ = std::make_unique<parallax::camera::CameraProducer>(*camera_, context_.products());
        
        if (lidar_) {
            lidar_producer_ = std::make_unique<parallax::lidar::RplidarSourceProducer>(*lidar_, context_.products());
        }
        
        isp_producer_ = std::make_unique<parallax::isp::IspProducer>(pipeline_.isp(), context_.products());

        rectification_producer_ = std::make_unique<parallax::stereo::RectificationProducer>(pipeline_.rectifier(),
                                                                                            pipeline_.calibration(),
                                                                                            context_.products());

        stereo_producer_ = std::make_unique<parallax::stereo::StereoProducer>(pipeline_.matcher(), context_.products());

        depth_producer_ = std::make_unique<parallax::stereo::DepthProducer>(pipeline_.calibration(),
                                                                            context_.products());

        charuco_pose_producer_ = std::make_unique<parallax::pose::CharucoPoseProducer>(
                                                  pipeline_.charucoPose(),
                                                  pipeline_.calibration(),
                                                  context_.products());

        marker_depth_producer_ = std::make_unique<parallax::pose::MarkerDepthPoducer>(context_.products());
        detection_producer_ = std::make_unique<parallax::perception::DetectionProducer>(*nanoowl_, context_.products());
        object3d_producer_ = std::make_unique<parallax::perception::Object3DProducer>(*stereo_roi_associator_,
                                                                                     *lidar_detection_associator_,
                                                                                     context_.products());
                                                                                     
        tracked_object3d_producer_ = std::make_unique<parallax::perception::TrackedObject3DProducer>(*stereo_roi_associator_, context_.products());

        segmentation_producer_ = std::make_unique<parallax::perception::SegmentationProducer>(*efficientvit_sam_,
                                                                                              context_.products(),
                                                                                              "models/efficientvit-sam/engines/l0_encoder_fp16.engine",
                                                                                              "models/efficientvit-sam/engines/l0_decoder_fp16.engine");
        segmented_depth_producer_ = std::make_unique<parallax::perception::SegmentedDepthProducer>(*stereo_roi_associator_, context_.products());

        single_target_producer_ = std::make_unique<parallax::tracking::SingleTargetProducer>(context_.products(), resolver_);
        
        cuvslam_producer_ = std::make_unique<parallax::localization::CuVslamProducer>(*cuvslam_localizer_, context_.products());

        spatial_map_ = std::make_unique<parallax::mapping::SpatialMap>(
            mapping_config_, context_.stereoLane().cudaHandle());

        spatial_tsdf_producer_ = std::make_unique<parallax::mapping::SpatialTsdfProducer>(
            pipeline_.calibration(), sensor_extrinsics_, mapping_config_,
            *spatial_map_, context_.products());

        tsdf_snapshot_producer_ = std::make_unique<parallax::mapping::TsdfSnapshotProducer>(
            mapping_config_, *spatial_map_, context_.products());

        esdf_producer_ = std::make_unique<parallax::mapping::EsdfProducer>(
            mapping_config_, *spatial_map_, context_.products());

        spatial_mesh_producer_ = std::make_unique<parallax::mapping::SpatialMeshProducer>(
            mapping_config_, pipeline_.calibration(), *spatial_map_, context_.products());

        localized_spatial_producer_ = std::make_unique<parallax::perception::LocalizedSpatialProducer>(pipeline_.calibration(),
                                                                                                       sensor_extrinsics_,
                                                                                                       context_.products());

        /**
         * Registration describes the complete concrete dependency graph.
         * Finalization happens exactly once, after every producer is present.
         */
        graph_.register_producer(*camera_producer_);
        graph_.register_producer(*isp_producer_);
        graph_.register_producer(*rectification_producer_);
        graph_.register_producer(*stereo_producer_);
        graph_.register_producer(*depth_producer_);
        graph_.register_producer(*charuco_pose_producer_);
        graph_.register_producer(*marker_depth_producer_);
        graph_.register_producer(*detection_producer_);
        graph_.register_producer(*single_target_producer_);

        if (lidar_producer_) graph_.register_producer(*lidar_producer_);
        graph_.register_producer(*object3d_producer_);
        graph_.register_producer(*tracked_object3d_producer_);
        graph_.register_producer(*localized_spatial_producer_);
        graph_.register_producer(*segmentation_producer_);
        graph_.register_producer(*segmented_depth_producer_);
        graph_.register_producer(*cuvslam_producer_);
        graph_.register_producer(*spatial_tsdf_producer_);
        graph_.register_producer(*tsdf_snapshot_producer_);
        graph_.register_producer(*esdf_producer_);
        graph_.register_producer(*spatial_mesh_producer_);

        graph_.finalize();

        configure_product_history(graph_, context_.products());

        /**
         * These are deliberately root products rather than an exhaustive list of
         * intermediates. DependencyResolver derives the required producer subgraph.
         *
         * RectifiedRgb keeps camera -> ISP -> rectification active.
         * Disparity extends that path through stereo matching.
         * MarkerDepth extends it through metric depth and marker pose/depth.
         * LidarScan keeps the independent RPLIDAR source active.
         */
        resolver_.acquire(ProductId::RectifiedRgb, DemandSource::RuntimeBaseline);
        resolver_.acquire(ProductId::Disparity, DemandSource::RuntimeBaseline);
        resolver_.acquire(ProductId::LocalizationOdometry, DemandSource::RuntimeBaseline);
        // Persistent mapping is baseline state. TSDF and mesh are derived,
        // demand-driven materializations and are not baseline products.
        resolver_.acquire(ProductId::SpatialMapState, DemandSource::RuntimeBaseline);
        if (lidar_producer_) resolver_.acquire(ProductId::LidarScan, DemandSource::RuntimeBaseline);


        parallax::visualization::FoxgloveServer::DemandCallbacks foxglove_demand;
        foxglove_demand.acquire = [this](ProductId product) {
                /**
                 * Subscription callbacks record external demand only.
                 * They deliberately do not resolve or submit the graph here.
                 */
                resolver_.acquire(product, DemandSource::FoxgloveSubscriber);
            };

        foxglove_demand.release = [this](ProductId product) { 
                                    resolver_.release(
                                    product,
                                    DemandSource::FoxgloveSubscriber);
            };


        foxglove::ServiceHandler command_handler {
            [this](const foxglove::ServiceRequest& request, foxglove::ServiceResponder&& responder) {
                /**
                 * Foxglove owns request/response transport.
                 *
                 * Runtime remains the composition boundary between transport,
                 * command decoding, and application request ownership.
                 *
                 * This callback performs bounded control-plane work only:
                 * decode -> validate -> mutate request/demand state -> respond.
                 * It never submits producers or waits for perception output.
                 */
                const auto parsed = parallax::application::parse_foxglove_command(request.payloadStr());

                if (!parsed.ok()) {
                    std::move(responder).respondError(parsed.message);
                    return;
                }

                const auto result = request_controller_.apply(parsed.command);

                if (result.applied() &&
                    parsed.command.verb ==
                        parallax::application::CommandVerb::NavigationGoal) {
                    const auto state = request_controller_.state();

                    auto goal =
                        std::make_shared<parallax::application::NavigationGoal>();
                    goal->position_m = state.navigation_goal_m;
                    goal->revision = state.navigation_goal_revision;

                    parallax::core::ProductMetadata metadata{};
                    metadata.timestamp =
                        parallax::core::ExecutionContext::now();
                    metadata.production_timestamp = metadata.timestamp;
                    metadata.valid = true;

                    std::shared_ptr<const parallax::application::NavigationGoal>
                        published = std::move(goal);

                    context_.products().publish(
                        parallax::core::make_product(
                            ProductId::NavigationGoal,
                            metadata,
                            std::move(published)));
                }

                if (result.applied() && foxglove_.requestStateChannel().hasSinks()) {

                    const auto state = request_controller_.state();

                    nlohmann::json request_state{{"marker_depth_requested", state.marker_depth_requested},
                                                 {"detection_requested", state.detection_requested},
                                                 {"detection_target", state.detection_target},
                                                 {"detection_query_revision", state.detection_query_revision},
                                                 {"tracking_requested", state.tracking_requested},
                                                 {"tracking_target", state.tracking_target},
                                                 {"tracking_query_revision", state.tracking_query_revision},
                                                 {"segmentation_requested", state.segmentation_requested},
                                                 {"segmentation_target", state.segmentation_target},
                                                 {"navigation_goal_requested", state.navigation_goal_requested},
                                                 {"navigation_goal", {
                                                     state.navigation_goal_m[0],
                                                     state.navigation_goal_m[1],
                                                     state.navigation_goal_m[2]}},
                                                 {"navigation_goal_revision", state.navigation_goal_revision}};

                    const std::string serialized_state = request_state.dump();
                    const auto error = foxglove_.requestStateChannel().log(reinterpret_cast<const std::byte*>(
                                                                           serialized_state.data()),
                                                                           serialized_state.size());

                    if (error != foxglove::FoxgloveError::Ok) {
                        std::cerr << "Runtime: request-state publication failed: " << foxglove::strerror(error) << '\n';
                    }
                }
                nlohmann::json response{{"accepted", true}, 
                                        {"status", result.applied() ? "applied" : "unavailable"},
                                        {"message", result.message}};

                const std::string serialized = response.dump();

                const auto* data = reinterpret_cast<const std::byte*>(serialized.data());

                std::vector<std::byte> payload{data, data + serialized.size()};
                std::move(responder).respondOk(payload);
            }
        };

        if (!foxglove_.initialize(std::move(foxglove_demand), std::move(command_handler))) {
            std::cerr << "Runtime: failed to initialize Foxglove server\n";
            shutdown();
            return false;
        }

        const auto& rgb = pipeline_.rgb();
        if (!publisher_.initialize(foxglove_, 
                                   rgb.width, 
                                   rgb.height, 
                                   config_.frame_rate, 
                                   sensor_extrinsics_.left_camera.child_frame,
                                   pipeline_.calibration())) {
            std::cerr << "Runtime: failed to initialize visualization publisher\n";

            shutdown();
            return false;
        }        
        initialized_ = true;
        return true;
    }
}

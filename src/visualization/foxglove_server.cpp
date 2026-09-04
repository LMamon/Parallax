#include <parallax/visualization/foxglove_server.hpp>

#include <foxglove/foxglove.hpp>
#include <foxglove/service.hpp>
#include <foxglove/schema.hpp>

#include <foxglove/service.hpp>

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace parallax::visualization {

    namespace {
        std::vector<std::byte> loadSchemaFile(const std::string& filename) {
            const std::string path = std::string(PARALLAX_SCHEMA_DIR) + "/" + filename;

            std::ifstream file(path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open Foxglove schema: " + path);
            }

            const std::string contents{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
            const auto* begin = reinterpret_cast<const std::byte*>(contents.data());

            return {begin, begin + contents.size()};
        }
    }

    FoxgloveServer::~FoxgloveServer() { shutdown(); }

    void FoxgloveServer::bindProduct(std::uint64_t channel_id, parallax::core::ProductId product) {
        product_by_channel_id_.emplace(channel_id, product);
    }

    std::optional<parallax::core::ProductId> FoxgloveServer::productForChannel(std::uint64_t channel_id) const noexcept {
        std::lock_guard<std::mutex> lock(subscription_mutex_);
        const auto it = product_by_channel_id_.find(channel_id);
        if (it == product_by_channel_id_.end()) {
            return std::nullopt;
        }

        return it->second;
    }

    void FoxgloveServer::onSubscribe(std::uint64_t channel_id, const foxglove::ClientMetadata&) {
        if (left_calibration_channel_ && channel_id == left_calibration_channel_->id()) {
            calibration_requested_.store(true);
            return;
        }

        if (transform_channel_ && channel_id == transform_channel_->id()) {
            transform_requested_.store(true);
            return;
        }
        
        std::optional<parallax::core::ProductId> product;
        bool acquire = false;

        {
            /**
             * Foxglove may invoke callbacks concurrently.
             *
             * Keep this critical section limited to channel lookup + subscriber
             * accounting. The demand callback is invoked after releasing this mutex
             * so visualization locks never nest around external application logic.
             */
            std::lock_guard<std::mutex> lock(subscription_mutex_);

            const auto binding = product_by_channel_id_.find(channel_id);
            if (binding == product_by_channel_id_.end()) {
                // Calibration and other non-graph channels intentionally carry
                // no ProductId demand.
                return;
            }

            auto& subscribers = subscriber_count_by_channel_[channel_id];
            acquire = (subscribers == 0);
            ++subscribers;
            product = binding->second;
        }

        if (acquire && product && demand_callbacks_.acquire) {
            demand_callbacks_.acquire(*product);
        }
    }

    void FoxgloveServer::onUnsubscribe(std::uint64_t channel_id, const foxglove::ClientMetadata&) {
        std::optional<parallax::core::ProductId> product;
        bool release = false;

        {
            std::lock_guard<std::mutex> lock(subscription_mutex_);

            const auto binding = product_by_channel_id_.find(channel_id);
            if (binding == product_by_channel_id_.end()) return;

            const auto count_it = subscriber_count_by_channel_.find(channel_id);

            if (count_it == subscriber_count_by_channel_.end() ||
                count_it->second == 0) {

                /**
                 * Defensive saturation.
                 *
                 * Foxglove documents balanced subscription callbacks, but an
                 * unexpected duplicate/unordered teardown notification must not
                 * underflow our size_t subscriber count.
                 */
                return;
            }

            --count_it->second;

            if (count_it->second == 0) {
                subscriber_count_by_channel_.erase(count_it);
                release = true;
            }
            product = binding->second;
        }

        if (release && product && demand_callbacks_.release) {
            demand_callbacks_.release(*product);
        }
    }

    void FoxgloveServer::releaseOutstandingDemand() {
        std::vector<parallax::core::ProductId> outstanding;
        {
            std::lock_guard<std::mutex> lock(subscription_mutex_);

            outstanding.reserve(
                subscriber_count_by_channel_.size());

            /**
             * Resolver receives one Foxglove-owned reference per channel on 0 -> 1,
             * not one reference per subscribing client. Therefore each still-active
             * channel needs exactly one compensating release here.
             */
            for (const auto& [channel_id, subscriber_count] : subscriber_count_by_channel_) {
                if (subscriber_count == 0) {
                    continue;
                }

                const auto binding = product_by_channel_id_.find(channel_id);
                if (binding != product_by_channel_id_.end()) {
                    outstanding.push_back(binding->second);
                }
            }

            subscriber_count_by_channel_.clear();
        }

        for (const auto product : outstanding) {
            if (demand_callbacks_.release) {
                demand_callbacks_.release(product);
            }
        }
    }

    bool FoxgloveServer::initializeChannels() {
        using parallax::core::ProductId;

        /**
         * Topic strings are defined exactly once here, at native Foxglove channel
         * creation. The channel objects themselves are the advertised capability
         * surface.
         *
         * Creating these channels must not acquire ProductId demand or execute a
         * producer. Subscription-driven demand is wired separately in Commit 2.
         */

        auto transforms = foxglove::messages::FrameTransformChannel::create("/tf", context_);
        if (!transforms.has_value()) {
            std::cerr << "Failed to create /tf channel: "
                      << foxglove::strerror(transforms.error()) << '\n';
            return false;
        }

        transform_channel_.emplace(std::move(transforms.value()));

        auto left_image = foxglove::messages::CompressedVideoChannel::create("/camera/left/image", context_);
        if (!left_image.has_value()) {
            std::cerr << "Failed to create /camera/left/image channel: "
                      << foxglove::strerror(left_image.error()) << '\n';
            return false;
        }

        left_image_channel_.emplace(std::move(left_image.value()));
        bindProduct(left_image_channel_->id(), ProductId::RectifiedRgb);

        auto calibration = foxglove::messages::CameraCalibrationChannel::create("/camera/left/calibration", context_);
        if (!calibration.has_value()) {
            std::cerr << "Failed to create /camera/left/calibration channel: "
                      << foxglove::strerror(calibration.error()) << '\n';
            return false;
        }

        left_calibration_channel_.emplace(std::move(calibration.value()));

        /**
         * Calibration is startup/static configuration rather than a graph product,
         * so it intentionally has no ProductId association.
         */
        auto disparity = foxglove::messages::RawImageChannel::create("/stereo/disparity", context_);
        if (!disparity.has_value()) {
            std::cerr << "Failed to create /stereo/disparity channel: "
                      << foxglove::strerror(disparity.error()) << '\n';
            return false;
        }

        disparity_channel_.emplace(std::move(disparity.value()));
        bindProduct(disparity_channel_->id(), ProductId::Disparity);

        auto depth = foxglove::messages::RawImageChannel::create("/stereo/depth", context_);
        if (!depth.has_value()) {
            std::cerr << "Failed to create /stereo/depth channel: "
                      << foxglove::strerror(depth.error()) << '\n';
            return false;
        }

        depth_channel_.emplace(std::move(depth.value()));
        bindProduct(depth_channel_->id(), ProductId::Depth);

        auto marker_pose = foxglove::messages::PoseInFrameChannel::create("/marker/pose", context_);
        if (!marker_pose.has_value()) {
            std::cerr << "Failed to create /marker/pose channel: "
                      << foxglove::strerror(marker_pose.error()) << '\n';
            return false;
        }

        marker_pose_channel_.emplace(std::move(marker_pose.value()));
        bindProduct(marker_pose_channel_->id(), ProductId::Pose);

        /**
         * MarkerDepth is a scalar measurement and does not have an appropriate
         * Foxglove well-known generated message type. RawChannel is Foxglove's
         * native extension mechanism for custom schemas; this keeps the transport,
         * schema advertisement, channel identity, and subscription behavior inside
         * the SDK rather than creating a Parallax protocol.
         */
        auto marker_depth = foxglove::RawChannel::create("/marker/depth", 
                                                         "json", 
                                                         foxglove::Schema{"parallax.MarkerDepth",
                                                         "jsonschema",
                                                         marker_depth_schema_.data(),
                                                         marker_depth_schema_.size()}, 
                                                         context_);

        if (!marker_depth.has_value()) {
            std::cerr << "Failed to create /marker/depth channel: "
                      << foxglove::strerror(marker_depth.error()) << '\n';
            return false;
        }

        marker_depth_channel_.emplace(std::move(marker_depth.value()));
        bindProduct(marker_depth_channel_->id(), ProductId::MarkerDepth);

        auto lidar_scan = foxglove::messages::LaserScanChannel::create("/lidar/scan", context_);
        if (!lidar_scan.has_value()) {
            std::cerr << "Failed to create /lidar/scan channel: "
                      << foxglove::strerror(lidar_scan.error()) << '\n';
            return false;
        }

        lidar_scan_channel_.emplace(std::move(lidar_scan.value()));
        bindProduct(lidar_scan_channel_->id(), ProductId::LidarScan);

        auto runtime_telemetry = foxglove::RawChannel::create("/parallax/runtime",
                                                              "json",
                                                              foxglove::Schema{"parallax.RuntimeTelemetry",
                                                              "jsonschema",
                                                              runtime_telemetry_schema_.data(),
                                                              runtime_telemetry_schema_.size()},
                                                              context_);

        if (!runtime_telemetry.has_value()) {
            std::cerr << "Failed to create /parallax/runtime channel: "
                      << foxglove::strerror(runtime_telemetry.error()) << '\n';

            return false;
        }
        runtime_telemetry_channel_.emplace(std::move(runtime_telemetry.value()));

        auto request_state = foxglove::RawChannel::create("/parallax/requests", 
                                                          "json", 
                                                          foxglove::Schema{"parallax.RequestState",
                                                          "jsonschema",
                                                          request_state_schema_.data(),
                                                          request_state_schema_.size()},
                                                          context_);

        if (!request_state.has_value()) {
            std::cerr << "Failed to create /parallax/requests channel: "
                      << foxglove::strerror(request_state.error()) << '\n';

            return false;
        }

        request_state_channel_.emplace(std::move(request_state.value()));
        auto detections = foxglove::RawChannel::create("/perception/detections",
                                                       "json",
                                                        foxglove::Schema{"parallax.DetectionSet",
                                                        "jsonschema",
                                                        detection_schema_.data(),
                                                        detection_schema_.size()},
                                                        context_);

        if (!detections.has_value()) {
            std::cerr << "Failed to create /perception/detections channel: "
                      << foxglove::strerror(detections.error()) << '\n';
            return false;
        }
        detection_channel_.emplace(std::move(detections.value()));
        bindProduct(detection_channel_->id(), ProductId::Detection);

        /**
         * Detection visualization uses Foxglove's native ImageAnnotations schema
         * rather than rasterizing rectangles into another full-frame image.
         *
         * Both detection channels observe the same compact ProductId::Detection.
         * Subscribing to either channel may therefore contribute normal
         * FoxgloveSubscriber demand without introducing a detector-specific
         * execution trigger.
         *
         * Coordinate-space note:
         * NanoOWL currently consumes RgbLeft while /camera/left/image is RectifiedRgb.
         * These annotations therefore describe the detector's source-image pixels.
         * Do not claim geometric alignment with the rectified image until a matching
         * image-space consumer or explicit coordinate transform is introduced.
         */
        auto detection_annotations = foxglove::messages::ImageAnnotationsChannel::create(
                                                                                "/perception/annotations",
                                                                                context_);

        if (!detection_annotations.has_value()) {
            std::cerr << "Failed to create /perception/annotations channel: "
                      << foxglove::strerror(detection_annotations.error()) << '\n';

            return false;
        }
        detection_annotations_channel_.emplace(std::move(detection_annotations.value()));
        bindProduct(detection_annotations_channel_->id(), ProductId::Detection);

        auto segmentation_mask = foxglove::messages::RawImageChannel::create("/perception/segmentation", context_);
        if (!segmentation_mask.has_value()) {
            std::cerr << "Failed to create /perception/segmentation channel: "
                      << foxglove::strerror(segmentation_mask.error()) << '\n';

            return false;
        }

        segmentation_mask_channel_.emplace(std::move(segmentation_mask.value()));
        bindProduct(segmentation_mask_channel_->id(), ProductId::Segmentation);

        auto track_annotations = foxglove::messages::ImageAnnotationsChannel::create("/tracking/annotations", context_);
        if (!track_annotations.has_value()) {
            std::cerr << "Failed to create /tracking/annotations channel: "
                      << foxglove::strerror(track_annotations.error()) << '\n';

            return false;
        }

        track_annotations_channel_.emplace(std::move(track_annotations.value()));
        bindProduct(track_annotations_channel_->id(), ProductId::Track2D);
        // every graph backed channel gets bindProduct(...)
        
        return true;
    }

    bool FoxgloveServer::initialize(DemandCallbacks demand_callbacks, CommandServiceHandler command_handler) {
        if (initialized_) return true;
        if (!demand_callbacks.valid()) {
            std::cerr << "Foxglove demand callbacks are not configured\n";
            return false;
        }
        demand_callbacks_ = std::move(demand_callbacks);
        foxglove::setLogLevel(foxglove::LogLevel::Info);

        /**
         * Use one explicit Foxglove Context for the Parallax server and all
         * advertised channels. Foxglove Context is the native binding between
         * channels and sinks.
         */
        context_ = foxglove::Context::create();

        try {
            marker_depth_schema_ = loadSchemaFile("marker_depth.json");
            runtime_telemetry_schema_ = loadSchemaFile("runtime_telemetry.json");
            command_request_schema_ = loadSchemaFile("command_request.json");
            command_response_schema_ = loadSchemaFile("command_response.json");
            request_state_schema_ = loadSchemaFile("request_state.json");
            detection_schema_ = loadSchemaFile("detections.json");
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';

            shutdown();
            return false;
        }

        foxglove::WebSocketServerOptions options{};
        options.context = context_;
        options.name = "Parallax";
        options.host = "0.0.0.0";
        options.port = 8765;
        options.message_backlog_size = 32;


        options.capabilities = foxglove::WebSocketServerCapabilities::Services;
        options.supported_encodings = {"json"};
        /**
        * Foxglove owns client/channel subscription semantics. Parallax observes
        * those native transitions rather than polling hasSinks() or maintaining
        * a second subscription protocol.
        *
        * SDK callbacks may run concurrently, so handlers below perform only
        * synchronized reference accounting and bounded demand notification.
        * They never serialize products or submit graph work.
        */

        options.callbacks.onSubscribe = [this](std::uint64_t channel_id, const foxglove::ClientMetadata& client) {
                onSubscribe(channel_id, client);
            };

        options.callbacks.onUnsubscribe = [this](std::uint64_t channel_id, const foxglove::ClientMetadata& client) {
                onUnsubscribe(channel_id, client);
            };

        auto result = foxglove::WebSocketServer::create(std::move(options));

        if (!result.has_value()) {
            std::cerr << "Failed to create Foxglove websocket server: "
                      << foxglove::strerror(result.error()) << '\n';

            return false;
        }

        server_ = std::make_unique<foxglove::WebSocketServer>(std::move(result.value()));

        /*
        * Commands are request/response operations rather than streamed products.
        *
        * Register the command service directly with the WebSocket server. The
        * service handler mutates application request/demand state only; producer
        * execution remains owned by Runtime and the dependency resolver.
        */
        foxglove::ServiceMessageSchema request_schema{"json", foxglove::Schema{"parallax.CommandRequest",
                                                                               "jsonschema",
                                                                               command_request_schema_.data(),
                                                                               command_request_schema_.size()}};

        foxglove::ServiceMessageSchema response_schema{"json", foxglove::Schema{"parallax.CommandResponse",
                                                                                "jsonschema",
                                                                                command_response_schema_.data(),
                                                                                command_response_schema_.size()}};

        foxglove::ServiceSchema service_schema{"parallax.Command", request_schema, response_schema};

        auto command_service = foxglove::Service::create("/parallax/command", service_schema, command_handler);
        if (!command_service.has_value()) {
            std::cerr << "Failed to create /parallax/command service: "
                      << foxglove::strerror(command_service.error()) << '\n';

            shutdown();
            return false;
        }

        const auto service_error = server_->addService(std::move(command_service.value()));
        if (service_error != foxglove::FoxgloveError::Ok) {
            std::cerr << "Failed to register /parallax/command service: "
                      << foxglove::strerror(service_error) << '\n';

            shutdown();
            return false;
        }

        if (!initializeChannels()) {
            shutdown();
            return false;
        }

        initialized_ = true;
        return true;
    }

    void FoxgloveServer::shutdown() {
        /**
         * Close advertised channels before stopping the server.
         *
         * This ordering matters once subscription callbacks are introduced:
         * Foxglove may report active channel unsubscriptions while the server is
         * still alive, allowing Parallax to release scoped demand cleanly.
         */
        if (left_image_channel_) {
            left_image_channel_->close();
            left_image_channel_.reset();
        }

        if (left_calibration_channel_) {
            left_calibration_channel_->close();
            left_calibration_channel_.reset();
        }

        if (disparity_channel_) {
            disparity_channel_->close();
            disparity_channel_.reset();
        }

        if (depth_channel_) {
            depth_channel_->close();
            depth_channel_.reset();
        }

        if (marker_pose_channel_) {
            marker_pose_channel_->close();
            marker_pose_channel_.reset();
        }

        if (marker_depth_channel_) {
            marker_depth_channel_->close();
            marker_depth_channel_.reset();
        }

        if (lidar_scan_channel_) {
            lidar_scan_channel_->close();
            lidar_scan_channel_.reset();
        }

        if (transform_channel_) {
            transform_channel_->close();
            transform_channel_.reset();
        }

        if (runtime_telemetry_channel_) {
            runtime_telemetry_channel_->close();
            runtime_telemetry_channel_.reset();
        }

        if (request_state_channel_) {
            request_state_channel_->close();
            request_state_channel_.reset();
        }

        if (detection_annotations_channel_) {
            detection_annotations_channel_->close();
            detection_annotations_channel_.reset();
        }

        if (detection_channel_) {
            detection_channel_->close();
            detection_channel_.reset();
        }

        if (server_) {
            server_->stop();
            server_.reset();
        }

        marker_depth_schema_.clear();
        runtime_telemetry_schema_.clear();
        command_request_schema_.clear();
        command_response_schema_.clear();
        request_state_schema_.clear();
        detection_schema_.clear();
        
        releaseOutstandingDemand();
        {
            std::lock_guard<std::mutex> lock(subscription_mutex_);
            product_by_channel_id_.clear();
        }

        demand_callbacks_ = {};
        initialized_ = false;
    }
}
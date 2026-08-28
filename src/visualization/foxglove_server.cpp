#include <parallax/visualization/foxglove_server.hpp>

#include <foxglove/foxglove.hpp>
#include <foxglove/schema.hpp>

#include <cstddef>
#include <iostream>
#include <string_view>
#include <utility>

namespace parallax::visualization {

    namespace {

        constexpr std::string_view kMarkerDepthSchema = R"json({
                                                            "$schema": "https://json-schema.org/draft/2020-12/schema",
                                                            "title": "parallax.MarkerDepth",
                                                            "type": "object",
                                                            "properties": {
                                                                "depth_m": {
                                                                    "type": "number"
                                                                },
                                                                "valid": {
                                                                    "type": "boolean"
                                                                }
                                                            },
                                                            "required": ["depth_m", "valid"],
                                                            "additionalProperties": false
                                                            }
                                                            )json";

        foxglove::Schema markerDepthSchema() {
            foxglove::Schema schema;
            schema.name = "parallax.MarkerDepth";
            schema.encoding = "jsonschema";
            schema.data = reinterpret_cast<const std::byte*>(kMarkerDepthSchema.data());
            schema.data_len = kMarkerDepthSchema.size();
            return schema;
        }

    }

    FoxgloveServer::~FoxgloveServer() { shutdown(); }

    void FoxgloveServer::bindProduct(std::uint64_t channel_id, parallax::core::ProductId product) {
        product_by_channel_id_.emplace(channel_id, product);
    }

    std::optional<parallax::core::ProductId> FoxgloveServer::productForChannel(std::uint64_t channel_id) const noexcept {
        const auto it = product_by_channel_id_.find(channel_id);
        if (it == product_by_channel_id_.end()) return std::nullopt;

        return it->second;
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

        auto left_image = foxglove::messages::CompressedVideoChannel::create( "/camera/left/image", context_);
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

        auto confidence = foxglove::messages::RawImageChannel::create("/stereo/confidence", context_);
        if (!confidence.has_value()) {
            std::cerr << "Failed to create /stereo/confidence channel: "
                      << foxglove::strerror(confidence.error()) << '\n';
            return false;
        }

        confidence_channel_.emplace(std::move(confidence.value()));
        bindProduct(confidence_channel_->id(), ProductId::Confidence);

        auto depth = foxglove::messages::RawImageChannel::create("/stereo/depth", context_);
        if (!depth.has_value()) {
            std::cerr
                << "Failed to create /stereo/depth channel: "
                << foxglove::strerror(depth.error())
                << '\n';
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
        auto marker_depth = foxglove::RawChannel::create("/marker/depth", "json", markerDepthSchema(), context_);
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

        return true;
    }

    bool FoxgloveServer::initialize() {
        if (initialized_) return true;
        foxglove::setLogLevel(foxglove::LogLevel::Info);

        /**
         * Use one explicit Foxglove Context for the Parallax server and all
         * advertised channels. Foxglove Context is the native binding between
         * channels and sinks.
         */
        context_ = foxglove::Context::create();

        foxglove::WebSocketServerOptions options{};
        options.context = context_;
        options.name = "Parallax";
        options.host = "0.0.0.0";
        options.port = 8765;

        /**
         * Do not install subscribe/unsubscribe callbacks here yet.
         * Commit 1 proves that capability advertisement is execution-neutral.
         * Commit 2 will attach native callbacks and translate their channel IDs
         * into scoped ProductId demand.
         */
        auto result = foxglove::WebSocketServer::create(std::move(options));

        if (!result.has_value()) {
            std::cerr << "Failed to create Foxglove websocket server: "
                      << foxglove::strerror(result.error()) << '\n';
            return false;
        }

        server_ = std::make_unique<foxglove::WebSocketServer>(std::move(result.value()));

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

        if (confidence_channel_) {
            confidence_channel_->close();
            confidence_channel_.reset();
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

        product_by_channel_id_.clear();

        if (server_) {
            server_->stop();
            server_.reset();
        }
        initialized_ = false;
    }
}
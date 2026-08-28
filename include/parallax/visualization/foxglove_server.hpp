#pragma once

#include <parallax/core/product_id.hpp>

#include <foxglove/channel.hpp>
#include <foxglove/context.hpp>
#include <foxglove/messages.hpp>
#include <foxglove/websocket.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

namespace parallax::visualization {

    /**
     * Owns the Foxglove observability surface.
     *
     * Foxglove channels are the authoritative topic/capability definitions.
     * Parallax does not maintain a second topic registry. The only graph-specific
     * association retained here is native Foxglove channel ID -> ProductId so
     * subscription callbacks can contribute graph demand in Phase 10.
     *
     * Channel existence advertises capability only. It does not imply that the
     * associated product is demanded, computed, ready, or currently publishing.
     */
    class FoxgloveServer {
        public:
            FoxgloveServer() = default;
            ~FoxgloveServer();

            FoxgloveServer(const FoxgloveServer&) = delete;
            FoxgloveServer& operator=(const FoxgloveServer&) = delete;

            bool initialize();
            void shutdown();

            [[nodiscard]] foxglove::WebSocketServer& server() noexcept {
                return *server_;
            }

            [[nodiscard]] bool initialized() const noexcept {
                return initialized_;
            }

            /**
             * Resolve a native Foxglove subscription identity into the graph
             * product represented by that channel.
             *
             * Calibration and any other non-graph configuration channels are
             * intentionally absent from this association.
             */
            [[nodiscard]]
            std::optional<parallax::core::ProductId>
            productForChannel(std::uint64_t channel_id) const noexcept;

            [[nodiscard]] foxglove::messages::CompressedVideoChannel& leftImageChannel() noexcept {
                return *left_image_channel_;
            }

            [[nodiscard]] foxglove::messages::CameraCalibrationChannel& leftCalibrationChannel() noexcept {
                return *left_calibration_channel_;
            }

            [[nodiscard]] foxglove::messages::RawImageChannel& disparityChannel() noexcept {
                return *disparity_channel_;
            }

            [[nodiscard]] foxglove::messages::RawImageChannel& confidenceChannel() noexcept {
                return *confidence_channel_;
            }

            [[nodiscard]] foxglove::messages::RawImageChannel& depthChannel() noexcept {
                return *depth_channel_;
            }

            [[nodiscard]] foxglove::messages::PoseInFrameChannel& markerPoseChannel() noexcept {
                return *marker_pose_channel_;
            }

            [[nodiscard]] foxglove::RawChannel& markerDepthChannel() noexcept {
                return *marker_depth_channel_;
            }

            [[nodiscard]] foxglove::messages::LaserScanChannel& lidarScanChannel() noexcept {
                return *lidar_scan_channel_;
            }

        private:
            bool initializeChannels();
            void bindProduct(std::uint64_t channel_id, parallax::core::ProductId product);

            /**
             * Dedicated Foxglove context for Parallax.
             *
             * A Context is Foxglove's native binding between channels and sinks.
             * The WebSocket server and every channel below use this same context,
             * avoiding an additional Parallax transport/registration layer.
             *
             * Declared before channels/server so it outlives them during normal
             * member destruction.
             */
            foxglove::Context context_{};

            std::unique_ptr<foxglove::WebSocketServer> server_;
            std::optional<foxglove::messages::CompressedVideoChannel> left_image_channel_;
            std::optional<foxglove::messages::CameraCalibrationChannel> left_calibration_channel_;
            std::optional<foxglove::messages::RawImageChannel> disparity_channel_;
            std::optional<foxglove::messages::RawImageChannel> confidence_channel_;
            std::optional<foxglove::messages::RawImageChannel> depth_channel_;
            std::optional<foxglove::messages::PoseInFrameChannel> marker_pose_channel_;

            /**
             * Foxglove does not provide a well-known scalar depth message whose
             * semantics match marker depth. Use Foxglove's native RawChannel +
             * schema facility rather than abusing an unrelated generated schema.
             */
            std::optional<foxglove::RawChannel> marker_depth_channel_;
            std::optional<foxglove::messages::LaserScanChannel> lidar_scan_channel_;

            /**
             * Runtime subscription identity only.
             *
             * Topic strings and schemas live exclusively in the Foxglove channel
             * definitions above; this map deliberately does not duplicate them.
             */
            std::unordered_map<std::uint64_t, parallax::core::ProductId> product_by_channel_id_;

            bool initialized_ = false;
    };
}
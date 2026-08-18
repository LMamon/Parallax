#pragma once

#include <foxglove/websocket.hpp>
#include <memory>

namespace parallax::visualization {
    class FoxgloveServer {
        public:
            FoxgloveServer() = default;
            ~FoxgloveServer();

            FoxgloveServer(const FoxgloveServer&) = delete;
            FoxgloveServer& operator=(const FoxgloveServer&) = delete;

            bool initialize();
            void shutdown();

        private:
            std::unique_ptr<foxglove::WebSocketServer> server_;
            bool initialized_ = false;
    };
}
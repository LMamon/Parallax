#include <parallax/visualization/foxglove_server.hpp>
#include <foxglove/foxglove.hpp>

#include <iostream>
#include <utility>
namespace parallax::visualization {
    FoxgloveServer::~FoxgloveServer() { shutdown(); }

    bool FoxgloveServer::initialize() {
        if (initialized_) return true;
        foxglove::setLogLevel(foxglove::LogLevel::Info);
        foxglove::WebSocketServerOptions options{};
        options.name = "Parallax";
        options.host = "0.0.0.0";
        options.port = 8765;

        auto result = foxglove::WebSocketServer::create(std::move(options));

        if (!result.has_value()) {
            std::cerr << "Failed to create Foxglove websocket server: " << foxglove::strerror(result.error())
                << '\n';
            return false;
        }

        server_ = std::make_unique<foxglove::WebSocketServer>(std::move(result.value()));
        initialized_ =true;
        return true;
    }

    void FoxgloveServer::shutdown() {
        if (!initialized_) return;
        if (server_) {
            server_->stop();
            server_.reset();
        }
        initialized_ = false;
    }
} // namespace parllax::visualization 

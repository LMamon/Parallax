#include <px_camera/logger.hpp>

#include <cerrno>
#include <cstring>
#include <iostream>

namespace px_camera {
    void logError(const char* operation) {
        std::cerr << operation
                  << " failed: "
                  << std::strerror(errno)
                  << '\n';
    }
    void logError(const char* operation, const char* object) {
        std::cerr << operation
              << " ("
              << object
              << ") failed: "
              << std::strerror(errno)
              << '\n';
    }
    void logMessage(const char* message) {
        std::cerr << message << '\n';
    }
}
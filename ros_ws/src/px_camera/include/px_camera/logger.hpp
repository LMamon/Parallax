#pragma once

namespace px_camera {
    void logError(const char* operation);
    void logError(const char* operation, const char* object);
    void logMessage(const char* message);
}
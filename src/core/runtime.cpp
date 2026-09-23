#include <parallax/core/runtime.hpp>

namespace parallax::core {
    Runtime::~Runtime() { shutdown(); }
}

#include <parallax/perception/nanoowl_bridge.hpp>

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <pybind11/embed.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

namespace py = pybind11;

namespace parallax::perception {

    class NanoOwlBridge::Impl {
        public:
            Impl() : interpreter{} {}

            py::scoped_interpreter interpreter;
            py::object detector;
    };

    NanoOwlBridge::NanoOwlBridge() = default;
    NanoOwlBridge::~NanoOwlBridge() { shutdown(); }

    bool NanoOwlBridge::initialize(
        const std::filesystem::path& engine_path) {
        if (initialized_) return true;

        if (!std::filesystem::is_regular_file(engine_path)) return false;

        try {
            impl_ = std::make_unique<Impl>();
            py::module_ module = py::module_::import("parallax.nanoowl_detector");
            py::object detector_type = module.attr("NanoOwlDetector");

            impl_->detector = detector_type(engine_path.string());

            initialized_ = true;
            return true;
        } catch (
            const py::error_already_set& error) {
            std::cerr << "NanoOWL initialization failed: " << error.what() << '\n';

            shutdown();
            return false;
        } catch (
            const std::exception& error) {
            std::cerr << "NanoOWL initialization failed: " << error.what() << '\n';

            shutdown();
            return false;
        }
    }

    bool NanoOwlBridge::setQuery(const std::string& query, std::uint64_t revision) {
        if (!initialized_ || !impl_) {
            return false;
        }

        try {
            py::gil_scoped_acquire gil;
            impl_->detector.attr("set_query")(query, revision);

            return true;
        } catch (
            const py::error_already_set& error) {
            std::cerr << "NanoOWL query failed: " << error.what() << '\n';
            return false;
        }
    }

    bool NanoOwlBridge::predict(const parallax::isp::StereoRgbFrame& frame, 
                                cudaStream_t stream,
                                DetectionSet& detections) {

        if (!initialized_ || !impl_ || stream == nullptr || !frame.left.isAllocated()) {
            return false;
        }

        if (frame.width == 0 || frame.height == 0 || frame.left.channels() != 3 || frame.left.elementSize() != 1) {
            return false;
        }

        try {
            py::gil_scoped_acquire gil;

            const auto torch_stream = c10::cuda::getStreamFromExternal(stream, 0);
            c10::cuda::CUDAStreamGuard stream_guard{torch_stream};

            const auto options = torch::TensorOptions{}.dtype(torch::kUInt8).device(
                                torch::Device{torch::kCUDA, 0}).requires_grad(false);

            const auto row_stride = static_cast<std::int64_t>(frame.left.pitch());

            constexpr std::int64_t pixel_stride = 3;
            constexpr std::int64_t channel_stride = 1;

            // from_blob does not own this CUDA allocation.
            // Product lifetime remains responsible for it.
            auto rgb = torch::from_blob(const_cast<void*>(frame.left.data()),
                                                        {
                                                            static_cast<std::int64_t>(frame.height),
                                                            static_cast<std::int64_t>(frame.width),
                                                            3
                                                        },
                                                        {
                                                            row_stride, pixel_stride, channel_stride
                                                        },
                                                        options
                                                    );

            // Torch's Python caster exposes the same tensor
            // storage to the existing NanoOWL adapter.
            py::object result = impl_->detector.attr("predict_rgb8")(rgb, 0.1F);
            detections = {};

            detections.query = result.attr("query").cast<std::string>();
            detections.query_revision =result.attr("query_revision").cast<std::uint64_t>();
            
            py::object output = result.attr("output");
            /*
            * NanoOWL decode output is intentionally materialized only at this
            * compact result boundary. Large image/tensor storage remains outside
            * the public DetectionSet contract.
            */
            torch::Tensor boxes = output.attr("boxes").cast<torch::Tensor>().to(torch::kCPU);
            torch::Tensor scores = output.attr("scores").cast<torch::Tensor>().to(torch::kCPU);
            torch::Tensor labels = output.attr("labels").cast<torch::Tensor>().to(torch::kCPU);

            boxes = boxes.contiguous();
            scores = scores.contiguous();
            labels = labels.contiguous();
            
            const auto count = boxes.size(0);
            detections.boxes.reserve(count);
            detections.scores.reserve(count);
            detections.labels.reserve(count);
            
            const auto boxes_access = boxes.accessor<float, 2>();
            const auto scores_access = scores.accessor<float, 1>();
            const auto labels_access = labels.accessor<std::int64_t, 1>();

            for (std::int64_t i = 0; i < count; ++i) {
                const float x0 = boxes_access[i][0];
                const float y0 = boxes_access[i][1];
                const float x1 = boxes_access[i][2];
                const float y1 = boxes_access[i][3];
   
                detections.boxes.emplace_back(x0, y0,
                                              x1 - x0,
                                              y1 - y0);
 
                detections.scores.push_back(scores_access[i]);
                detections.labels.push_back(labels_access[i]);
            }

            return detections.valid();
        } catch (const py::error_already_set& error) {
            std::cerr << "NanoOWL prediction failed: " << error.what() << '\n';
            return false;

        } catch (const c10::Error& error) {
            std::cerr << "Torch CUDA bridge failed: " << error.what() << '\n';
            return false;
        }
    }

    void NanoOwlBridge::shutdown() noexcept {
        if (!impl_) {
            initialized_ = false;
            return;
        }

        try {
            py::gil_scoped_acquire gil;

            if (!impl_->detector.is_none() && impl_->detector) {
                impl_->detector.attr("close")();
            }
        } catch (...) {
            // Shutdown must remain noexcept.
        }

        impl_.reset();
        initialized_ = false;
    }
}
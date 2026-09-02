#include <parallax/perception/efficientvit_sam.hpp>

#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace parallax::perception {

    namespace {

    class TensorRtLogger final : public nvinfer1::ILogger {
        public:
            void log(Severity severity, const char* message) noexcept override {
                if (severity <= Severity::kWARNING) {
                    std::cerr << "[TensorRT] " << message << '\n';
                }
            }
        };

        TensorRtLogger g_logger;

        std::vector<char> read_file(const std::filesystem::path& path) {
            std::ifstream file(path, std::ios::binary | std::ios::ate);

            if (!file) {
                throw std::runtime_error("failed to open TensorRT engine: " + path.string());
            }

            const auto size = file.tellg();

            if (size <= 0) {
                throw std::runtime_error("TensorRT engine is empty: " + path.string());
            }

            std::vector<char> data(static_cast<std::size_t>(size));

            file.seekg(0, std::ios::beg);

            if (!file.read(data.data(), size)) {
                throw std::runtime_error("failed to read TensorRT engine: " + path.string());
            }
            return data;
        }

        bool has_tensors(const nvinfer1::ICudaEngine& engine, const std::unordered_set<std::string>& expected) {
            std::unordered_set<std::string> found;

            for (int i = 0; i < engine.getNbIOTensors(); ++i) {
                found.emplace(engine.getIOTensorName(i));
            }

            for (const auto& name : expected) {
                if (found.find(name) == found.end()) {
                    std::cerr << "TensorRT engine missing expected tensor: " << name << '\n';
                    return false;
                }
            }

            return true;
        }

        }


    class EfficientVitSam::Impl {
        public:
            ~Impl() { shutdown(); }

            bool initialize(const std::filesystem::path& encoder_path, const std::filesystem::path& decoder_path) {
                shutdown();

                try {
                    runtime_.reset(nvinfer1::createInferRuntime(g_logger));

                    if (!runtime_) {
                        throw std::runtime_error("failed to create TensorRT runtime");
                    }

                    const auto encoder_data = read_file(encoder_path);
                    const auto decoder_data = read_file(decoder_path);

                    encoder_.reset(runtime_->deserializeCudaEngine(
                                   encoder_data.data(),
                                   encoder_data.size()));

                    if (!encoder_) {
                        throw std::runtime_error("failed to deserialize encoder engine");
                    }

                    decoder_.reset(
                        runtime_->deserializeCudaEngine(decoder_data.data(),
                                                        decoder_data.size()));

                    if (!decoder_) {
                        throw std::runtime_error("failed to deserialize decoder engine");
                    }

                    if (!has_tensors(*encoder_, {"input_image", "image_embeddings"})) {
                        throw std::runtime_error("unexpected encoder tensor contract");
                    }

                    if (!has_tensors(*decoder_,
                                    {"image_embeddings",
                                    "point_coords",
                                    "point_labels",
                                    "masks",
                                    "iou_predictions"})) {
                                
                        throw std::runtime_error("unexpected decoder tensor contract");
                    }

                    encoder_context_.reset(encoder_->createExecutionContext());
                    decoder_context_.reset(decoder_->createExecutionContext());
                    if (!encoder_context_ || !decoder_context_) {
                        throw std::runtime_error("failed to create TensorRT execution contexts");
                    }

                    return true;
                }
                catch (const std::exception& error) {
                    std::cerr << "EfficientVitSam initialization failed: " << error.what() << '\n';

                    shutdown();
                    return false;
                }
            }

            void shutdown() noexcept {
                decoder_context_.reset();
                encoder_context_.reset();

                decoder_.reset();
                encoder_.reset();

                runtime_.reset();

                metrics_ = {};
            }

            EfficientVitSamMetrics metrics() const noexcept { return metrics_; }

        private:
            std::unique_ptr<nvinfer1::IRuntime> runtime_;

            std::unique_ptr<nvinfer1::ICudaEngine> encoder_;
            std::unique_ptr<nvinfer1::ICudaEngine> decoder_;

            std::unique_ptr<nvinfer1::IExecutionContext> encoder_context_;
            std::unique_ptr<nvinfer1::IExecutionContext> decoder_context_;

            EfficientVitSamMetrics metrics_{};
        };


        EfficientVitSam::EfficientVitSam() : impl_(std::make_unique<Impl>()) {}
        EfficientVitSam::~EfficientVitSam() { shutdown(); }


        bool EfficientVitSam::initialize(const std::filesystem::path& encoder_engine,
                                         const std::filesystem::path& decoder_engine) {

            if (initialized_) return true;

            initialized_ = impl_->initialize(encoder_engine, decoder_engine);
            return initialized_;
        }


        bool EfficientVitSam::segment(const parallax::isp::StereoRgbFrame&,
                                      const cv::Rect2f&,
                                      cudaStream_t,
                                      EfficientVitSamResult&) {
            // Inference is added after engine lifecycle and tensor
            // contracts have been validated independently.
            return false;
        }


        void EfficientVitSam::shutdown() noexcept {
            if (impl_) impl_->shutdown();
            initialized_ = false;
        }

        EfficientVitSamMetrics EfficientVitSam::metrics() const noexcept {
            if (!impl_) return {};
            return impl_->metrics();
        }
    }
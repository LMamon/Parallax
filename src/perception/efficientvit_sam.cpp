#include <parallax/perception/efficientvit_sam.hpp>
#include <parallax/perception/efficientvit_sam_cuda.hpp>
#include <parallax/core/fixed_payload_pool.hpp>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

        bool dims_equal(const nvinfer1::Dims& dims, std::initializer_list<std::int64_t> expected) {
            if (dims.nbDims != static_cast<int>(expected.size())) {
                return false;
            }

            int index = 0;

            for (const auto dimension : expected) {
                if (dims.d[index] != dimension) return false;

                ++index;
            }

            return true;
        }

        void require_tensor(const nvinfer1::ICudaEngine& engine,
                            const char* name,
                            nvinfer1::TensorIOMode mode,
                            nvinfer1::DataType type,
                            std::initializer_list<std::int64_t> dimensions) {

            if (engine.getTensorIOMode(name) != mode) {
                throw std::runtime_error(std::string("unexpected TensorRT I/O mode for tensor: ") + name);
            }

            if (engine.getTensorDataType(name) != type) {
                throw std::runtime_error(std::string("unexpected TensorRT data type for tensor: ") + name);
            }

            if (!dims_equal(engine.getTensorShape(name), dimensions)) {
                throw std::runtime_error(std::string("unexpected TensorRT shape for tensor: ") + name);
            }
        }

        void validate_encoder(const nvinfer1::ICudaEngine& engine) {
            if (!has_tensors(engine, {"input_image", "image_embeddings"} )) {
                throw std::runtime_error("unexpected encoder tensor contract");
            }

            require_tensor(engine,
                           "input_image",
                           nvinfer1::TensorIOMode::kINPUT,
                           nvinfer1::DataType::kFLOAT,
                           {1, 3, 512, 512});

            require_tensor(engine,
                           "image_embeddings",
                           nvinfer1::TensorIOMode::kOUTPUT,
                           nvinfer1::DataType::kFLOAT,
                           {1, 256, 64, 64});
        }

        void validate_decoder(const nvinfer1::ICudaEngine& engine) {
            if (!has_tensors(engine, {"image_embeddings",
                                      "point_coords",
                                      "point_labels",
                                      "masks",
                                      "iou_predictions"})) {

                throw std::runtime_error("unexpected decoder tensor contract");
            }

            require_tensor(engine,
                           "image_embeddings",
                           nvinfer1::TensorIOMode::kINPUT,
                           nvinfer1::DataType::kFLOAT,
                           {1, 256, 64, 64});

            require_tensor(engine,
                           "point_coords",
                           nvinfer1::TensorIOMode::kINPUT,
                           nvinfer1::DataType::kFLOAT,
                           {1, 2, 2});

            require_tensor(engine,
                           "point_labels",
                           nvinfer1::TensorIOMode::kINPUT,
                           nvinfer1::DataType::kFLOAT,
                           {1, 2});

            require_tensor(engine,
                           "masks",
                           nvinfer1::TensorIOMode::kOUTPUT,
                           nvinfer1::DataType::kFLOAT,
                           {1, 1, 256, 256});

            require_tensor(engine,
                           "iou_predictions",
                           nvinfer1::TensorIOMode::kOUTPUT,
                           nvinfer1::DataType::kFLOAT,
                           {1, 1});
        }

        EfficientVitSamGeometry make_geometry(std::uint32_t width, std::uint32_t height) {
            const auto longest = static_cast<float>(std::max(width, height));

            const float encoder_scale = 512.0F / longest;
            const float prompt_scale = 1024.0F / longest;

            EfficientVitSamGeometry geometry{};

            geometry.original_width = static_cast<int>(width);
            geometry.original_height = static_cast<int>(height);

            geometry.encoder_width = static_cast<int>(static_cast<float>(width) * encoder_scale + 0.5F);
            geometry.encoder_height = static_cast<int>(static_cast<float>(height) * encoder_scale + 0.5F);

            geometry.prompt_width = static_cast<int>(static_cast<float>(width) * prompt_scale + 0.5F);
            geometry.prompt_height = static_cast<int>(static_cast<float>(height) * prompt_scale + 0.5F);

            return geometry;
        }

        constexpr std::size_t float_bytes(std::size_t count) {
            return count * sizeof(float);
        }

        bool bind_tensor(nvinfer1::IExecutionContext& context, const char* name, void* address) {
            if (!address) return false;

            if (!context.setTensorAddress(name, address)) {
                std::cerr << "failed to bind TensorRT tensor: " << name << '\n';
                return false;
            }
            return true;
        }
    }

    class EfficientVitSam::Impl {
        private:
            class DeviceBuffer {
                public:
                    DeviceBuffer() = default;
                    ~DeviceBuffer() { release(); }

                    DeviceBuffer(const DeviceBuffer&) = delete;
                    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

                    bool allocate(std::size_t bytes) {
                        release();

                        if (bytes == 0) return false;

                        if (cudaMalloc(&data_, bytes) != cudaSuccess) {
                            data_ = nullptr;
                            bytes_ = 0;
                            return false;
                        }

                        bytes_ = bytes;
                        return true;
                    }

                    void release() noexcept {
                        if (data_) cudaFree(data_);

                        data_ = nullptr;
                        bytes_ = 0;
                    }

                    [[nodiscard]] void* data() noexcept { return data_; }
                    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

                private:
                    void* data_ = nullptr;
                    std::size_t bytes_ = 0;
            };

            struct MaskSlot {
                DeviceBuffer buffer;
                std::uint32_t width = 0;
                std::uint32_t height = 0;
            };

            static constexpr std::size_t MaskSlotCount = 2;

            static constexpr std::size_t PersistentBufferBytes = float_bytes(3ULL * 512ULL * 512ULL) +
                                                                 float_bytes(256ULL * 64ULL * 64ULL) +
                                                                 float_bytes(4) +
                                                                 float_bytes(2) +
                                                                 float_bytes(256ULL * 256ULL) +
                                                                 float_bytes(1);

        public:
            ~Impl() { shutdown(); }

            bool initialize(const std::filesystem::path& encoder_path, const std::filesystem::path& decoder_path) {
                shutdown();

                try {
                    runtime_.reset(nvinfer1::createInferRuntime(g_logger));

                    if (!runtime_) throw std::runtime_error("failed to create TensorRT runtime");

                    const auto encoder_data = read_file(encoder_path);
                    const auto decoder_data = read_file(decoder_path);

                    encoder_.reset(runtime_->deserializeCudaEngine(encoder_data.data(), encoder_data.size()));
                    if (!encoder_) throw std::runtime_error("failed to deserialize encoder engine");

                    decoder_.reset(runtime_->deserializeCudaEngine(decoder_data.data(), decoder_data.size()));
                    if (!decoder_) throw std::runtime_error("failed to deserialize decoder engine");

                    validate_encoder(*encoder_);
                    validate_decoder(*decoder_);

                    encoder_context_.reset(encoder_->createExecutionContext());
                    decoder_context_.reset(decoder_->createExecutionContext());

                    if (!encoder_context_ || !decoder_context_) {
                        throw std::runtime_error("failed to create TensorRT execution contexts");
                    }

                    if (!encoder_context_->setInputShape("input_image", nvinfer1::Dims4{1, 3, 512, 512})) {
                        throw std::runtime_error("failed to configure encoder input shape");
                    }

                    if (!decoder_context_->setInputShape("point_coords", nvinfer1::Dims3{1, 2, 2}) ||
                        !decoder_context_->setInputShape("point_labels", nvinfer1::Dims2{1, 2})) {

                        throw std::runtime_error("failed to configure decoder prompt shapes");
                    }

                    if (!allocate_buffers()) throw std::runtime_error("failed to allocate EfficientViT-SAM device buffers");
                    if (!bind_buffers()) throw std::runtime_error("failed to bind EfficientViT-SAM device buffers");

                    metrics_.model_loaded = true;
                    ++metrics_.model_load_count;
                    metrics_.persistent_buffer_bytes = PersistentBufferBytes;
                    metrics_.mask_pool_capacity = MaskSlotCount;

                    return true;
                }
                catch (const std::exception& error) {
                    std::cerr << "EfficientVitSam initialization failed: " << error.what() << '\n';
                    shutdown();
                    return false;
                }
            }

            void shutdown() noexcept {
                mask_pool_.reset();
                decoder_context_.reset();
                encoder_context_.reset();

                decoder_.reset();
                encoder_.reset();

                runtime_.reset();

                metrics_.model_loaded = false;
                metrics_.mask_pool_in_use = 0;
            }

            bool allocate_buffers() {
                return encoder_input_.allocate(float_bytes(3ULL * 512ULL * 512ULL)) &&
                       image_embeddings_.allocate(float_bytes(256ULL * 64ULL * 64ULL)) &&
                       point_coords_.allocate(float_bytes(4)) &&
                       point_labels_.allocate(float_bytes(2)) &&
                       low_res_mask_.allocate(float_bytes(256ULL * 256ULL)) &&
                       iou_prediction_.allocate(float_bytes(1));
            }

            bool bind_buffers() {
                return bind_tensor(*encoder_context_, "input_image", encoder_input_.data()) &&
                       bind_tensor(*encoder_context_, "image_embeddings", image_embeddings_.data()) &&
                       bind_tensor(*decoder_context_, "image_embeddings", image_embeddings_.data()) &&
                       bind_tensor(*decoder_context_, "point_coords", point_coords_.data()) &&
                       bind_tensor(*decoder_context_, "point_labels", point_labels_.data()) &&
                       bind_tensor(*decoder_context_, "masks", low_res_mask_.data()) &&
                       bind_tensor(*decoder_context_, "iou_predictions", iou_prediction_.data());
            }

            bool initialize_mask_pool(std::uint32_t width, std::uint32_t height) {
                if (mask_pool_.initialized()) {
                    const auto* prototype = mask_pool_.prototype();
                    return prototype && prototype->width == width && prototype->height == height;
                }

                const std::size_t bytes =static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

                return mask_pool_.initialize([width, height, bytes](MaskSlot& slot, std::size_t) {
                        if (!slot.buffer.allocate(bytes)) return false;

                        slot.width = width;
                        slot.height = height;

                        return true;
                    });
            }

            std::shared_ptr<MaskSlot> acquire_mask(std::uint32_t width, std::uint32_t height) {
                if (!initialize_mask_pool(width, height)) return {};
                return mask_pool_.acquire();
            }

            bool execute_encoder(cudaStream_t stream) { return encoder_context_ && encoder_context_->enqueueV3(stream); }
            bool execute_decoder(cudaStream_t stream) { return decoder_context_ && decoder_context_->enqueueV3(stream); }
            float* low_res_mask() noexcept { return static_cast<float*>(low_res_mask_.data()); }
            void record_inference() noexcept { ++metrics_.inference_count; }

            EfficientVitSamMetrics metrics() const noexcept {
                auto metrics = metrics_;

                metrics.mask_pool_in_use = mask_pool_.in_use();
                metrics.mask_pool_capacity = mask_pool_.capacity();
                metrics.mask_pool_high_water = mask_pool_.high_water_mark();

                return metrics;
            }

            void* encoder_input() noexcept { return encoder_input_.data(); }
            void* point_coords() noexcept { return point_coords_.data(); }
            void* point_labels() noexcept { return point_labels_.data(); }

        private:
            std::unique_ptr<nvinfer1::IRuntime> runtime_;
            
            std::unique_ptr<nvinfer1::IRuntime> decoder_runtime_;
            std::unique_ptr<nvinfer1::IRuntime> encoder_runtime_;

            std::unique_ptr<nvinfer1::ICudaEngine> encoder_;
            std::unique_ptr<nvinfer1::ICudaEngine> decoder_;
            
            std::unique_ptr<nvinfer1::IExecutionContext> encoder_context_;
            std::unique_ptr<nvinfer1::IExecutionContext> decoder_context_;
            
            EfficientVitSamMetrics metrics_{};
            
            DeviceBuffer encoder_input_;
            DeviceBuffer image_embeddings_;
            
            DeviceBuffer point_coords_;
            DeviceBuffer point_labels_;
            
            DeviceBuffer low_res_mask_;
            DeviceBuffer iou_prediction_;

            parallax::core::FixedPayloadPool<MaskSlot, MaskSlotCount> mask_pool_;
    };

    EfficientVitSam::EfficientVitSam() : impl_(std::make_unique<Impl>()) {}

    EfficientVitSam::~EfficientVitSam() { shutdown(); }
    bool EfficientVitSam::initialize(const std::filesystem::path& encoder_engine, const std::filesystem::path& decoder_engine) {
        if (initialized_) return true;
        initialized_ = impl_->initialize(encoder_engine, decoder_engine);

        return initialized_;
    }

    bool EfficientVitSam::segment(const parallax::isp::StereoRgbFrame& frame, 
                                  const cv::Rect2f& box, cudaStream_t stream,
                                  EfficientVitSamResult& result) {

        result = {};
        if (!initialized_ || !impl_ || stream == nullptr || !frame.left.isAllocated()) {
            return false;
        }

        if (frame.width == 0 || frame.height == 0 || frame.left.channels() != 3 || frame.left.elementSize() != 1) {
            return false;
        }

        if (box.width <= 0.0F || box.height <= 0.0F) {
            return false;
        }

        const auto geometry = make_geometry(frame.width, frame.height);
        if (geometry.encoder_width <= 0 || geometry.encoder_height <= 0 || geometry.prompt_width <= 0 || geometry.prompt_height <= 0) {
            return false;
        }

        if (!preprocess_efficientvit_sam(static_cast<const std::uint8_t*>(frame.left.data()),
                                         frame.left.pitch(),
                                         geometry.original_width,
                                         geometry.original_height,
                                         static_cast<float*>(impl_->encoder_input()),
                                         geometry.encoder_width,
                                         geometry.encoder_height,
                                         stream)) {

            return false;
        }

        if (!prepare_efficientvit_sam_box(box.x,
                                          box.y,
                                          box.x + box.width,
                                          box.y + box.height,
                                          geometry.original_width,
                                          geometry.original_height,
                                          geometry.prompt_width,
                                          geometry.prompt_height,
                                          static_cast<float*>(impl_->point_coords()),
                                          static_cast<float*>(impl_->point_labels()),
                                          stream)) {

            return false;
        }
        if (!impl_->execute_encoder(stream)) return false;
        if (!impl_->execute_decoder(stream)) return false;

        auto mask = impl_->acquire_mask(frame.width, frame.height);

        if (!mask) return false;
        if (!postprocess_efficientvit_sam_mask(impl_->low_res_mask(),
                                               geometry.prompt_width,
                                               geometry.prompt_height,
                                               static_cast<std::uint8_t*>(mask->buffer.data()),
                                               geometry.original_width,
                                               geometry.original_height,
                                               stream)) {

            return false;
        }

        const void* mask_data = mask->buffer.data();
        /*
        * The alias keeps the fixed pool slot leased while exposing only the
        * CUDA mask allocation to downstream products.
        */
        result.storage = std::shared_ptr<const void>(mask, mask_data);

        result.width = frame.width;
        result.height = frame.height;
        result.pitch_bytes = frame.width;
        result.confidence = 0.0F;

        impl_->record_inference();

        return result.valid();
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
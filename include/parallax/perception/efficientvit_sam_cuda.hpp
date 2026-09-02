#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace parallax::perception {

    bool preprocess_efficientvit_sam(const std::uint8_t* input,
                                     std::size_t input_pitch_bytes,
                                     int input_width,
                                     int input_height,
                                     float* output,
                                     int output_width,
                                     int output_height,
                                     cudaStream_t stream);

    bool prepare_efficientvit_sam_box(float x0,
                                      float y0,
                                      float x1,
                                      float y1,
                                      int original_width,
                                      int original_height,
                                      int prompt_width,
                                      int prompt_height,
                                      float* point_coords,
                                      float* point_labels,
                                      cudaStream_t stream);

}
#pragma once

#include "datastructures/mml_array.hpp"

namespace NodeUtils {

inline void compute_pool_attributes(std::string& auto_pad,
                                    std::vector<int>& kernel_shape,
                                    std::vector<int>& strides,
                                    std::vector<int>& pads,
                                    std::vector<int>& dilations) {
  const size_t spatial_rank = kernel_shape.size();

  if (strides.empty()) {
    strides = std::vector<int>(spatial_rank, 1);
  }

  if (pads.empty() && auto_pad == "NOTSET") {
    pads = std::vector<int>(spatial_rank * 2, 0);  // begin and end
  }

  if (dilations.empty()) {
    dilations = std::vector<int>(spatial_rank, 1);
  }
}

// Follows onnx specifications but onnxruntime does it differently.
inline array_mml<size_t> compute_pool_output_shape(
    const array_mml<size_t>& input_shape, const std::string& auto_pad,
    const int ceil_mode, const std::vector<int>& dilations,
    const std::vector<int>& kernel_shape, const std::vector<int>& pads,
    const std::vector<int>& strides) {
  size_t spacial_rank = kernel_shape.size();
  std::vector<size_t> output_shape_vector = {input_shape[0], input_shape[1]};

  for (size_t i = 0; i < spacial_rank; ++i) {
    int input_dim = input_shape[i + 2];
    int kernel = kernel_shape[i];
    int stride = strides[i];
    int dilation = dilations[i];

    int effective_kernel = (kernel - 1) * dilation + 1;

    size_t out_dim;
    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
      if (ceil_mode) {
        out_dim = std::ceil(static_cast<float>(input_dim) / stride);
      } else {
        out_dim = std::floor(static_cast<float>((input_dim - 1) / stride)) + 1;
      }
    } else if (auto_pad == "VALID") {
      if (ceil_mode) {
        out_dim = std::ceil(
            static_cast<float>(input_dim - effective_kernel + 1) / stride);
      } else {
        out_dim = std::floor(static_cast<float>(input_dim - effective_kernel) /
                             stride) +
                  1;
      }
    } else {
      int total_pad = pads[i] + pads[i + spacial_rank];
      if (ceil_mode) {
        out_dim = std::ceil(
            static_cast<float>(input_dim + total_pad - effective_kernel) /
                stride +
            1);
      } else {
        out_dim = std::floor(
            static_cast<float>(input_dim + total_pad - effective_kernel) /
                stride +
            1);
      }
    }

    output_shape_vector.push_back(out_dim);
  }
  return array_mml<size_t>(output_shape_vector);
}

// Follows onnx specifications but onnxruntime does it differently.
inline std::vector<std::pair<int, int>> compute_pool_pad_begin_end(
    const array_mml<size_t>& input_shape, const std::string& auto_pad,
    const int ceil_mode, const std::vector<int>& dilations,
    const std::vector<int>& kernel_shape, const std::vector<int>& pads,
    const std::vector<int>& strides) {
  size_t spatial_rank = kernel_shape.size();
  std::vector<std::pair<int, int>> pad_pairs(spatial_rank);

  for (size_t i = 0; i < spatial_rank; ++i) {
    int input = input_shape[i + 2];
    int kernel = kernel_shape[i];
    int stride = strides[i];
    int dilation = dilations[i];

    int effective_kernel = (kernel - 1) * dilation + 1;

    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
      float out_dim =
          ceil_mode ? std::ceil(static_cast<float>(input) / stride)
                    : std::floor(static_cast<float>(input - 1) / stride) + 1;

      int total_pad = std::max(0, static_cast<int>((out_dim - 1) * stride +
                                                   effective_kernel - input));

      int pad_begin =
          (auto_pad == "SAME_LOWER") ? (total_pad + 1) / 2 : total_pad / 2;
      int pad_end = total_pad - pad_begin;

      pad_pairs[i] = {pad_begin, pad_end};

    } else if (auto_pad == "VALID") {
      pad_pairs[i] = {0, 0};
    } else {
      pad_pairs[i] = {pads[i], pads[i + spatial_rank]};
    }
  }

  return pad_pairs;
}

}  // namespace NodeUtils
#pragma once

#include <functional>
#include <memory>

#include "datastructures/a_tensor.hpp"
#include "datastructures/tensor_concept.hpp"

/// @brief Tensor Operations Function Types
namespace toft {
/**
 * @typedef gemm_func
 * @brief Function signature for general matrix multiplication operations
 *
 * @tparam T The numeric type of the tensor elements
 */
template <TensorConcept::Types T>
using gemm_func = std::function<void(
    int TA, int TB, int M, int N, int K, T ALPHA, std::shared_ptr<Tensor<T>> A,
    int lda, std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
    std::shared_ptr<Tensor<T>> C, int ldc)>;

/**
 * @typedef gemm_onnx_func
 * @brief Function signature for ONNX-style general matrix multiplication
 *
 * @tparam T The numeric type of the tensor elements
 */
template <TensorConcept::Types T>
using gemm_onnx_func = std::function<std::shared_ptr<Tensor<T>>(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C)>;

/**
 * @typedef add_func
 * @brief Function signature for element-wise addition operations
 *
 * @tparam T The numeric type of the tensor elements
 */
template <TensorConcept::Types T>
using add_func = std::function<void(const std::shared_ptr<const Tensor<T>> a,
                                    const std::shared_ptr<const Tensor<T>> b,
                                    std::shared_ptr<Tensor<T>> c)>;

/**
 * @typedef subtract_func
 * @brief Function signature for element-wise subtraction operations
 *
 * @tparam T The numeric type of the tensor elements
 */
template <TensorConcept::Types T>
using subtract_func = std::function<void(const std::shared_ptr<Tensor<T>> a,
                                         const std::shared_ptr<Tensor<T>> b,
                                         std::shared_ptr<Tensor<T>> c)>;

/**
 * @typedef multiply_func
 * @brief Function signature for element-wise multiplication operations
 *
 * @tparam T The numeric type of the tensor elements
 */
template <TensorConcept::Types T>
using multiply_func =
    std::function<void(const std::shared_ptr<Tensor<T>> a, const T b,
                       std::shared_ptr<Tensor<T>> c)>;

/**
 * @typedef equals_func
 * @brief Function signature for element-wise equality comparison
 *
 * @tparam T The numeric type of the tensor elements
 */
template <TensorConcept::Types T>
using equals_func = std::function<bool(const std::shared_ptr<Tensor<T>> a,
                                       const std::shared_ptr<Tensor<T>> b)>;

/**
 * @typedef elementwise_func
 * @brief Function signature for general element-wise operations with output
 *
 * @tparam T The numeric type of the tensor elements
 */
template <TensorConcept::Types T>
using elementwise_func = std::function<void(
    const std::shared_ptr<const Tensor<T>> a, const std::function<T(T)>& f,
    const std::shared_ptr<Tensor<T>> c)>;

/**
 * @typedef elementwise_in_place_func
 * @brief Function signature for in-place element-wise operations
 *
 * @tparam T The numeric type of the tensor elements
 */
template <TensorConcept::Types T>
using elementwise_in_place_func = std::function<void(
    const std::shared_ptr<Tensor<T>> a, const std::function<T(T)>& f)>;

/**
 * @typedef arg_max_func
 * @brief Function signature for argmax operations
 *
 * @tparam T The numeric type of the tensor elements
 */
template <TensorConcept::Types T>
using arg_max_func =
    std::function<int(const std::shared_ptr<const Tensor<T>> a)>;

/**
 * @typedef sliding_window_func
 * @brief Function signature for sliding window operations (e.g., pooling)
 *
 * @tparam T The numeric type of the tensor elements
 */
template <TensorConcept::Types T>
using sliding_window_func = std::function<void(
    const array_mml<size_t>& in_shape, const array_mml<size_t>& out_shape,
    const std::vector<int>& kernel_shape, const std::vector<int>& strides,
    const std::vector<int>& dilations,
    const std::vector<std::pair<int, int>>& pads,
    const std::function<void(const std::vector<std::vector<size_t>>&,
                             const std::vector<size_t>&)>& window_f)>;

}  // namespace toft
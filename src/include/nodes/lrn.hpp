#pragma once

#include <stddef.h>

#include <string>
#include <variant>

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

/**
 * @class LRNNode_mml
 * @brief Performs Local Response Normalization
 * @details LRNNode_mml performs Local Response Normalization according to the
 * ONNX specifications. It normalizes the tensor across local input regions. The
 * local region is defined across the channels.
 */
class LRNNode_mml : public Node {
 public:
  /**
   * @typedef T
   * @brief Type alias for supported floating-point types in LRN operations
   */
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for LRNNode_mml
   *
   * @param X The name/key of the input tensor
   * @param Y The name/key of the output tensor
   * @param size The number of channels to sum over. Must be at least 1
   * @param alpha Scaling parameter. Default = 0.0001
   * @param beta The exponent. Must be at least 0. Default = 0.75
   * @param bias Bias to avoid division with 0. Must be at least 0.001. Default
   * = 1.0
   * @throws std::invalid_argument If size < 1 or bias < 0.001
   */
  LRNNode_mml(const std::string &X, const std::string &Y, size_t size,
              float alpha = 0.0001f, float beta = 0.75f, float bias = 1.0f);

  /**
   * @brief Constructor for LRNNode_mml from JSON
   *
   * @param node JSON object representing the LRN node
   */
  explicit LRNNode_mml(const nlohmann::json &node);

  /**
   * @brief Performs the forward pass computation for Local Response
   * Normalization
   *
   * @param iomap Map containing input and output tensors indexed by name
   */
  void forward(
      std::unordered_map<std::string, GeneralDataTypes> &iomap) override;

  /**
   * @brief Get inputs.
   *
   * @return The names of the inputs to the node.
   */
  std::vector<std::string> getInputs() override;

  /**
   * @brief Get outputs.
   *
   * @return The names of the outputs to the node.
   */
  std::vector<std::string> getOutputs() override;

 private:
  ///@brief Shared pointer to the input tensor
  std::string X;

  ///@brief Shared pointer to the output tensor
  std::string Y;

  ///@brief Scaling parameter
  float alpha;

  ///@brief The exponent
  float beta;

  ///@brief To avoid division by zero
  float bias;

  ///@brief Number of channels to sum over
  size_t size;
};

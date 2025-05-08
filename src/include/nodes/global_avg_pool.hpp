#pragma once

#include <string>
#include <variant>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

/**
 * @class GlobalAvgPoolNode
 * @brief A class representing a Global Average Pooling node in a computational
 * graph.
 *
 * This node performs global average pooling on the input tensor, reducing
 * spatial dimensions to 1x1 while computing the average of all values in each
 * feature map. It's commonly used in convolutional neural networks to reduce
 * spatial dimensions before fully connected layers.
 */
class GlobalAvgPoolNode : public Node {
 public:
  /**
   * @typedef T
   * @brief Type alias for supported numeric types in global average pooling
   * operations
   */
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for GlobalAvgPoolNode.
   *
   * @param X Shared pointer to the input tensor.
   * @param Y Shared pointer to the output tensor.
   */
  GlobalAvgPoolNode(const std::string &X, const std::string &Y);

  /**
   * @brief Constructor for GlobalAvgPoolNode from JSON.
   *
   * @param node JSON object representing the GlobalAveragePool node.
   */
  explicit GlobalAvgPoolNode(const nlohmann::json &node);

  /**
   * @brief Perform the forward pass computation of global average pooling.
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
  std::string X;  // Input tensor X.
  std::string Y;  // Output tensor Y.
};
#pragma once

#include <stdint.h>

#include <string>
#include <variant>

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

/**
 * @class AddNode
 * @brief A class representing an addition operation node in a computational
 * graph.
 *
 * This class inherits from the Node class and represents the element-wise
 * addition of two tensors in a computational graph. It performs the forward
 * pass computation by adding the elements of two input tensors and storing the
 * result in an output tensor. It supports broadcasting for tensors with
 * compatible shapes.
 */
class AddNode : public Node {
 public:
  /**
   * @typedef T
   * @brief Type alias for supported numeric types in addition operations
   */
  using T = std::variant<double, float, int32_t, int64_t>;

  /**
   * @typedef TensorT
   * @brief Type alias for shared pointers to tensors with supported types
   *
   * Gets std::variant<std::shared_ptr<tensor<T>>, ...> from T
   */
  using TensorT = TensorVariant<T>;

  /**
   * @brief Constructor for AddNode.
   *
   * @param A String hash ID to the first input tensor.
   * @param B String hash ID to the second input tensor.
   * @param C String hash ID to the output tensor.
   */
  AddNode(const std::string &A, const std::string &B, const std::string &C);

  /**
   * @brief Constructor for AddNode from JSON.
   *
   * @param node JSON object representing the Add node.
   */
  explicit AddNode(const nlohmann::json &node);

  /**
   * @brief Performs element-wise binary addition in the two input tensors and
   * stores the result in the output tensor.
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
  // tensors
  std::string A;  // Input tensor A
  std::string B;  // Input tensor B
  std::string C;  // Output tensor C

  /**
   * @brief Helper std::function used when broadcasting addition is required.
   * Likely only temporary to be replaced with something that can be used in
   * multiple nodes instead.
   *
   * @param a_ptr Shared pointer to the first input tensor
   * @param b_ptr Shared pointer to the second input tensor
   * @param c_ptr Shared pointer to the output tensor
   */
  void broadcast_addition(const TensorT &a_ptr, const TensorT &b_ptr,
                          const TensorT &c_ptr) const;
};

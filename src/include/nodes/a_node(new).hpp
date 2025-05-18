#pragma once
#include <concepts>
#include <nlohmann/json.hpp>

#include "tensor_concept.hpp"

template <TensorConcept::Types InputType, TensorConcept::Types OutputType>
class Node_new {
 public:
  virtual ~Node_new() = default;

  virtual Tensor<OutputType> forward(const Tensor<InputType>& input) = 0;
};

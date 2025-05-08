#include "nodes/dropout.hpp"

#include <map>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "nlohmann/json.hpp"

DropoutNode::DropoutNode(const std::string &data, const std::string &output,
                         const std::optional<std::string> &mask, float ratio,
                         bool training_mode, std::optional<int> seed)
    : data(data),
      output(output),
      mask(mask),
      ratio(ratio),
      training_mode(training_mode),
      seed(seed) {}

DropoutNode::DropoutNode(const nlohmann::json &node) {
  if (node.contains("input") && node["input"].is_array()) {
    data = node["input"][0];
  }

  if (node.contains("output") && node["output"].is_array()) {
    output = node["output"][0];
    if (node["output"].size() > 1) {
      mask = node["output"][1];
    }
  }

  ratio = 0.5;
  training_mode = false;
  seed = std::nullopt;
  if (node.contains("attribute") && node["attribute"].is_array()) {
    for (const auto &attr : node["attribute"]) {
      if (attr["name"] == "ratio") {
        ratio = attr["f"];
      } else if (attr["name"] == "training_mode") {
        // training_mode;
      } else if (attr["name"] == "seed") {
        seed = std::stoi(attr["i"].get<std::string>());
      }
    }
  }
}

void DropoutNode::forward(
    std::unordered_map<std::string, GeneralDataTypes> &iomap) {
  auto data_it = iomap.find(data);
  if (data_it == iomap.end()) {
    throw std::runtime_error(
        "ReshapeNode: Input tensor data not found in iomap");
  }

  const GeneralDataTypes &data_tensor = data_it->second;

  std::visit(
      [&](const auto &data_ptr) {
        using TensorPtr = std::decay_t<decltype(data_ptr)>;
        using TensorType = typename TensorPtr::element_type;
        using ValueType = typename TensorType::value_type;

        if constexpr (!is_in_variant_v<ValueType, T>) {
          throw std::runtime_error(
              "DropoutNode: Unsupported data type for tensor data");
        } else {
          auto output_it = iomap.find(output);
          if (output_it == iomap.end()) {
            // Create a new output tensor by copying the input tensor
            auto output_ptr = data_ptr->copy();
            // No need to fill with zeros as the dropout std::function will
            // overwrite the values
            iomap[output] = output_ptr;
            output_it = iomap.find(output);
          } else if (!std::holds_alternative<
                         std::shared_ptr<Tensor<ValueType>>>(
                         output_it->second)) {
            throw std::runtime_error(
                "DropoutNode: Output tensor has incorrect type");
          }

          auto output_ptr =
              std::get<std::shared_ptr<Tensor<ValueType>>>(output_it->second);

          if (data_ptr->get_shape().size() < 1) {
            throw std::runtime_error("Tensor data must be at least 1D.");
          }

          if (training_mode) {
            throw std::runtime_error(
                "DropoutNode forward pass in training mode is "
                "not implemented yet.");
          } else {
            *output_ptr = *data_ptr;
          }
        }
      },
      data_tensor);
}

std::vector<std::string> DropoutNode::getInputs() { return {data}; }

std::vector<std::string> DropoutNode::getOutputs() {
  if (mask.has_value()) {
    return {output, mask.value()};
  } else {
    return {output};
  }
}
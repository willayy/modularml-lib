# ModularML
Lightweight and modular library for building cpp integrated machine learning models from onnx models.

### Installation

Install the library and link it to your project using CMake. This is done by pasting the following code into your CMakeLists.txt file.

```cmake
# ----------------------- MODULARML -------------------------- #
include(FetchContent)
FetchContent_Declare(
    modularml
    GIT_REPOSITORY https://github.com/willayy/modularml-lib
    GIT_TAG        <use latest release tag>
)
FetchContent_MakeAvailable(modularml)
target_link_libraries(MyProject PRIVATE modularml)
# ------------------------------------------------------------ #
```

### Usage
```cpp
#include <modularml>
#include <memory>
#include <iostream>

// Define the blocked gemm flag to enable the compilation of the blocked gemm function
#define USE_BLOCKED_GEMM

int main() {

  // Dynamically change the gemm function pointer to point to the blocked version, this works for any function 
  // with a matching signature.
  TensorOperations::set_gemm_ptr<int, float, double>(
    mml_gemm_blocked<int>, 
    mml_gemm_blocked<float>, 
    mml_gemm_blocked<double>
  );

  // Use the TensorFactory to create tensors
  auto a = TensorFactory::create_tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = TensorFactory::create_tensor<int>({3, 2}, {7, 8, 9, 10, 11, 12});
  auto c = TensorFactory::create_tensor<int>({2, 3});

  // Perform the gemm operation using the blocked version.
  TensorOperations::gemm<int>(0, 0, 2, 2, 3, 1, a, 3, b, 2, 1, c, 2);

  // Print the result, ensuring that the computation works.
  std::cout << "Result of gemm: " << (*c) << std::endl;

  return 0;
}
```

#### Usage with AlexNet
```cpp
#include <modularml>
#include <memory>
#include <iostream>
#include <fstream>

int main() {
  /* Load the json file that has been created by onnx2json, that script can be downloaded externally
  Or you can use the one thats found in build/_deps/modularml-src/onnx2json */
  std::ifstream file("../src/alexnet_trained.json");
  nlohmann::json json_model;
  file >> json_model;
  file.close();
  // Parse the model into a model object, containing all the nodes.
  auto parser = Parser_mml();
  auto model = parser.parse(json_model);
  // Create an input tensor thats empty
  auto input = TensorFactory::create_tensor<float>({1, 3, 224, 224});
  // Create an input map
  std::unordered_map<std::string, GeneralDataTypes> input_map;
  input_map["input"] = input;
  // Run inference with the input map
  auto output_map = model->infer(input_map);
  // Get the output
  auto output = std::get_if<std::shared_ptr<Tensor<float>>>(&(output_map["output"]));
  // Extract the output tensor and print it!
  std::cout << "Output: " << (*output) << std::endl;
  return 0;
}
```

### Contributing
We welcome contributions!  
Please read our [Contributing Guide](CONTRIBUTING.md) for instructions on how to get started.

### License
This project is licensed under the [MIT License](LICENSE).

### Acknowledgments
This library is a fork from the framework that was developed as part of a Bachelor Thesis at Chalmers University of Technology. Which can be found [here](https://github.com/willayy/modularml)
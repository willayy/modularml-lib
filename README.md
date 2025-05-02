# ModularML
Lightweight and modular library for building cpp integrated machine learning models from onnx models.

### Installation

Install the library and link it to your project using CMake.

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

### Contributing
We welcome contributions!  
Please read our [Contributing Guide](CONTRIBUTING.md) for instructions on how to get started.

### License
This project is licensed under the [MIT License](LICENSE).

### Acknowledgments
This library is a fork from the framework that was developed as part of a Bachelor Thesis at Chalmers University of Technology. Which can be found [here](https://github.com/willayy/modularml)
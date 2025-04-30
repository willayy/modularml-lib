#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "datastructures/array_utility.hpp"

template <typename T>
array_mml<T> generate_random_array_mml_integral(size_t lo_sz, size_t hi_sz,
                                                T lo_v, T hi_v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(lo_sz, hi_sz);
  size_t n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);
  std::uniform_int_distribution<T> int_dist(lo_v, hi_v);
  for (size_t i = 0; i < n; i++) {
    arr[i] = int_dist(gen);
  }
  return arr;
}

template <typename T>
array_mml<T> generate_random_array_mml_real(size_t lo_sz, size_t hi_sz, T lo_v,
                                            T hi_v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(lo_sz, hi_sz);
  size_t n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);
  std::uniform_real_distribution<T> real_dist(lo_v, hi_v);
  for (size_t i = 0; i < n; i++) {
    arr[i] = real_dist(gen);
  }
  return arr;
}
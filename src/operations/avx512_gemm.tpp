#include "datastructures/tensor_concept.hpp"
#include "operations/avx512_gemm.hpp"

#if defined(USE_AVX_GEMM) || defined(USE_AVX512_GEMM)
#include <immintrin.h>
#endif

#ifdef USE_AVX512_GEMM
template <TensorConcept::Types T>
static void mml_gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                            std::shared_ptr<Tensor<T>> A, int lda,
                            std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                            std::shared_ptr<Tensor<T>> C, int ldc) {
  if (TA == 1)
    throw std::invalid_argument(
        "Transpose A not yet supported for AVX-512 GEMM.");
  if (TB == 1)
    throw std::invalid_argument(
        "Transpose B not yet supported for AVX-512 GEMM.");

  if constexpr (std::is_same<T, float>::value) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j += 16) {
        __m512 c_val = _mm512_loadu_ps(&(*C)[i * ldc + j]);
        __m512 sum = _mm512_setzero_ps();

        for (int k = 0; k < K; k++) {
          __m512 a_vals = _mm512_set1_ps((*A)[i * lda + k]);

          __m512 b_vals = _mm512_loadu_ps(&(*B)[k * ldb + j]);

          sum = _mm512_fmadd_ps(a_vals, b_vals, sum);
        }

        sum = _mm512_fmadd_ps(_mm512_set1_ps(ALPHA), sum, c_val);
        sum = _mm512_add_ps(sum, _mm512_set1_ps(BETA));

        _mm512_storeu_ps(&(*C)[i * ldc + j], sum);
      }
    }
  } else if constexpr (std::is_same<T, double>::value) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j += 8) {
        __m512d c_val = _mm512_loadu_pd(&(*C)[i * ldc + j]);
        __m512d sum = _mm512_setzero_pd();

        for (int k = 0; k < K; k++) {
          __m512d a_vals = _mm512_set1_pd((*A)[i * lda + k]);

          __m512d b_vals = _mm512_loadu_pd(&(*B)[k * ldb + j]);

          sum = _mm512_fmadd_pd(a_vals, b_vals, sum);
        }

        sum = _mm512_fmadd_pd(_mm512_set1_pd(ALPHA), sum, c_val);
        sum = _mm512_add_pd(sum, _mm512_set1_pd(BETA));

        _mm512_storeu_pd(&(*C)[i * ldc + j], sum);
      }
    }
  } else if constexpr (std::is_same<T, int>::value) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j += 16) {
        __m512i sum = _mm512_setzero_si512();

        for (int k = 0; k < K; ++k) {
          __m512i a_vals =
              _mm512_set1_epi32((*A)[i * lda + k]);  // scalar broadcast

          __m512i b_vals = _mm512_loadu_si512(
              reinterpret_cast<const void *>(&(*B)[k * ldb + j]));

          __m512i product = _mm512_mullo_epi32(a_vals, b_vals);
          sum = _mm512_add_epi32(sum, product);
        }

        sum = _mm512_mullo_epi32(_mm512_set1_epi32(ALPHA), sum);
        sum = _mm512_add_epi32(sum, _mm512_set1_epi32(BETA));

        _mm512_storeu_si512(reinterpret_cast<void *>(&(*C)[i * ldc + j]), sum);
      }
    }
  } else {
    throw std::runtime_error("AVX-512 only suppports double, float or int");
  }
}
#endif
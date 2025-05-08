#include "datastructures/tensor_concept.hpp"
#include "operations/intel_mkl_gemm.hpp"

#if defined(USE_INTEL_MKL_GEMM)
/**
 * @brief Performs general matrix multiplication using Intel MKL
 *
 * This function template provides an interface to Intel's Math Kernel Library
 * (MKL) for high-performance general matrix multiplication (GEMM) operations.
 * The operation performed is: C = alpha*op(A)*op(B) + beta*C
 * where op(X) is either X or X^T (transpose of X) based on the TA/TB
 * parameters.
 *
 * @note This functionality is not yet implemented and will throw an exception
 * if called.
 *
 * @tparam T Numeric data type that satisfies TensorConcept::Types
 * @param TA Transpose flag for matrix A (0: no transpose, 1: transpose)
 * @param TB Transpose flag for matrix B (0: no transpose, 1: transpose)
 * @param M Number of rows in op(A) and C
 * @param N Number of columns in op(B) and C
 * @param K Number of columns in op(A) and rows in op(B)
 * @param ALPHA Scalar multiplier for the product of matrices A and B
 * @param A Shared pointer to the first input tensor
 * @param lda Leading dimension of A
 * @param B Shared pointer to the second input tensor
 * @param ldb Leading dimension of B
 * @param BETA Scalar multiplier for matrix C
 * @param C Shared pointer to the output tensor
 * @param ldc Leading dimension of C
 * @throws std::invalid_argument Always thrown since this function is not yet
 * implemented
 */
template <TensorConcept::Types T>
static void mml_gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                               std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                               std::shared_ptr<Tensor<T>> C, int ldc) {
  std::invalid_argument("Intel MKL GEMM not yet supported.");
}
#endif
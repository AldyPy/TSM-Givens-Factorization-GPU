# Givens Factorization on GPU for Tall and Skinny Matrices (TSM)

## General Information
This repository contains source code for the implementation of the algorithm I have proposed in my undergrad thesis. The implementation is of an asymptotically optimal algorithm for Givens QR by Modi and Clarke (1984). 

## Abstract
*This abstract is machine translated, but has been proof-read.*

In the field of scientific computing, one problem that continues to receive active research attention is the QR decomposition of a matrix. Within this problem lies a subproblem: QR decomposition of tall-and-skinny matrices (matrices whose number of rows is significantly larger than the number of columns). This subproblem is an important component in several computational applications, such as the least squares problem and stationary video background subtraction.

A standard algorithm for QR decomposition is the Householder method. This method has been extensively studied in the literature, and its memory-optimal variant, Block Householder, is available in the NVIDIA CUDA toolkit, specifically through the cuSOLVER library. For general-purpose applications, the QR decomposition implementation provided by this library can be considered state of the art. However, for the subproblem of QR decomposition of tall-and-skinny matrices, although the topic has been studied in the literature, existing work employs a wide variety of optimization techniques, and there is no definitive algorithm that is widely accepted as the best approach.

This study implements and evaluates an alternative QR decomposition algorithm known as the Givens Rotation method. Existing literature generally considers this method unsuitable for general QR decomposition because of its primary drawback: a large number of sequential dependencies. However, for tall-and-skinny matrices, the parallelization scheme employed in this study is able to exploit the algorithm's inherent parallelism more effectively, thereby substantially mitigating this limitation. The resulting implementation outperformed the cuSOLVER library on the NVIDIA H100 for matrices with a height of 10⁶ (one million) rows, but remained slower than cuSOLVER on the RTX 4090 across all tested matrix sizes. Analysis and evaluation of the implementation indicate that it exhibits a favorable scaling factor with respect to matrix height. In other words, as the ratio of matrix height to matrix width increases, the relative performance of the proposed algorithm improves compared with the general-purpose QR decomposition algorithm provided by the cuSOLVER library.

## Notes
- It should be noted that NVIDIA does not actually disclose the implementation of cuSOLVER's `geqrf` kernel, other than that it is a variant of the Householder algorithm. So *technically*, the claim that cuSOLVER uses Blocked Householder can't be verified. However, it is assumed that a widely used, optimized kernel for general-purpose QR should implement a memory-optimized variant of Householder, of which Blocked Householder is the most standard and popular.

- The parallel schema/algorithm written by Modi and Clarke is commonly referred to as "Greedy", and have been independently discovered by multiple authors in classic literature. However, Modi and Clarke have presented a proper analysis of the algorithm's complexity and their findings have been among the most cited.

## Further Notes

This `README.md` is probably incomplete and will be expanded upon in the future.
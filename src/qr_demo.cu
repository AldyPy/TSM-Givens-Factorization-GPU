#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "matrix_ops.h"


int main(int argc, char* argv[]) {

    // if (argc != 2) {
    //     printf("Usage: %s <matrix_file>\n", argv[0]);
    //     return -1;
    // }

    float *A;
    int M, N;       // dimensions of A
    read_matrix(argv[1], &M, &N, &A);
    
    float *B = (float*) malloc(sizeof(float) * M * N);
    matmul_f(A, A, B, M, M, M);

    // print_matrix(A, M, N);
    // print_matrix(A, M, N);
    // print_matrix(B, M, N);


    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);


    // // create solver
    // cusolverDnHandle_t solver;
    // cusolverDnCreate(&solver);

    // float *d_A, *d_tau;
    // int *devInfo;

    // // fill A
    // for (int i = 0; i < M*N; i++) A[i] = (float)(rand() % 10);

    // cudaMalloc((void**)&d_A, sizeof(float)*M*N);
    // cudaMalloc((void**)&d_tau, sizeof(float)*(M < N ? M : N));
    // cudaMalloc((void**)&devInfo, sizeof(int));

    // cudaMemcpy(d_A, A, sizeof(float)*M*N, cudaMemcpyHostToDevice);

    // int lwork = 0;
    // cusolverDnSgeqrf_bufferSize(solver, M, N, d_A, M, &lwork);

    // float *work;
    // cudaMalloc((void**)&work, sizeof(float)*lwork);

    // // QR factorization
    // cusolverDnSgeqrf(
    //     solver,
    //     M, N,
    //     d_A, M,
    //     d_tau,
    //     work, lwork,
    //     devInfo
    // );


    // cudaMemcpy(A, d_A, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    // printf("R + Householder stored in A:\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%6.2f ", A[i*N + j]);
    //     }
    //     printf("\n");
    // }

    // cudaFree(d_A);
    // cudaFree(d_tau);
    // cudaFree(work);
    // cudaFree(devInfo);
    // cusolverDnDestroy(solver);

    // return 0;
}
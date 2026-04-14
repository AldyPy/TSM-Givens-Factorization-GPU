#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "matrix_ops.h"
#include "givens.h"

int verbose = 0;

// Source - https://stackoverflow.com/a/14038590
// Posted by talonmies, modified by community. See post 'Timeline' for change history
// Retrieved 2026-04-14, License - CC BY-SA 4.0
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


int main(int argc, char* argv[]) {

    if (argc > 2) verbose = 1;

    float *A;
    size_t M, N;       // dimensions of A
    read_matrix(argv[1], &M, &N, &A);

    printf("I/O done.\n");

    
    float* R = (float*) malloc (sizeof(float) * M * N);
    memcpy(R, A, M*N*sizeof(float));
    float* Q = (float*) malloc (sizeof(float) * M * M);
    memset(Q, 0, M*M*sizeof(float));

    for (int i = 0; i < M; i++) Q[i*M + i] = 1;

    if (verbose) print_matrix(A, M, N);

    if (verbose) printf("-------------------\n");


    int threads = 256;
    int blocks = (M*M + M*N + 255) / 256;
    size_t* leftmost = (size_t*) malloc (sizeof(size_t) * M);
    size_t* downmost = (size_t*) malloc (sizeof(size_t) * N);
    for (int i = 0; i < M; i++) leftmost[i] = 0;
    for (int i = 0; i < N; i++) downmost[i] = M - 1;

    size_t* leftmost_d ;
    size_t* downmost_d ;
    gpuErrCheck( cudaMalloc(&leftmost_d, sizeof(size_t) * M) );
    gpuErrCheck( cudaMalloc(&downmost_d, sizeof(size_t) * N) );
    gpuErrCheck( cudaMemcpy(leftmost_d, leftmost, M*sizeof(size_t), cudaMemcpyHostToDevice) );
    gpuErrCheck( cudaMemcpy(downmost_d, downmost, N*sizeof(size_t), cudaMemcpyHostToDevice) );


    float* R_d;
    float* Q_d;
    gpuErrCheck( cudaMalloc(&R_d, M*N*sizeof(float)) );
    gpuErrCheck( cudaMalloc(&Q_d, M*M*sizeof(float)) );
    gpuErrCheck( cudaMemcpy(R_d, R, M*N*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrCheck( cudaMemcpy(Q_d, Q, M*M*sizeof(float), cudaMemcpyHostToDevice) );

    printf("Malloc and memcpy done, calling Kernel...\n");


    for (int i = 0; i < M; i++) printf("%lld ", (unsigned long long)leftmost[i]);
    printf("\n");

    givens_gpu<<<blocks, threads>>>(
        Q_d, R_d, M, N, leftmost_d, downmost_d
    );
    
    gpuErrCheck( cudaPeekAtLastError() );
    gpuErrCheck( cudaDeviceSynchronize() );

    gpuErrCheck( cudaMemcpy(R, R_d, M*N*sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrCheck( cudaMemcpy(Q, Q_d, M*M*sizeof(float), cudaMemcpyDeviceToHost) );

    // givens_factorization(R, M, N, Q);
    
    if (verbose) print_matrix(R, M, N);
    if (verbose) print_matrix(Q, M, M);

    float* reconstructed_A = (float*) malloc (sizeof(float) * M * N);
    matmul_f(Q, R, reconstructed_A, M, M, N);

    if (verbose) print_matrix(reconstructed_A, M, N);
    printf("Error: %.3f\n", max_err(A, reconstructed_A, M, N));

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
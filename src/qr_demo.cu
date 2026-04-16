#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "matrix_ops.h"
#include "givens.h"

int verbose = 0;
int check = 0;

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

    if (argc > 2) check = 1;
    if (argc > 3) verbose = 1;

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
    // size_t* leftmost = (size_t*) malloc (sizeof(size_t) * M);
    // size_t* downmost = (size_t*) malloc (sizeof(size_t) * N);
    size_t* leftmost;
    size_t* downmost;
    cudaMallocManaged(&leftmost, sizeof(size_t)*M);
    cudaMallocManaged(&downmost, sizeof(size_t)*N);
    for (int i = 0; i < M; i++) leftmost[i] = 0;
    for (int i = 0; i < N; i++) downmost[i] = M - 1;

    // size_t* leftmost_d ;
    // size_t* downmost_d ;
    // gpuErrCheck( cudaMalloc(&leftmost_d, sizeof(size_t) * M) );
    // gpuErrCheck( cudaMalloc(&downmost_d, sizeof(size_t) * N) );
    // gpuErrCheck( cudaMemcpy(leftmost_d, leftmost, M*sizeof(size_t), cudaMemcpyHostToDevice) );
    // gpuErrCheck( cudaMemcpy(downmost_d, downmost, N*sizeof(size_t), cudaMemcpyHostToDevice) );


    float* R1_d;
    float* Q1_d;
    float* R2_d;
    float* Q2_d;
    gpuErrCheck( cudaMalloc(&R1_d, M*N*sizeof(float)) );
    gpuErrCheck( cudaMalloc(&Q1_d, M*M*sizeof(float)) );
    gpuErrCheck( cudaMalloc(&R2_d, M*N*sizeof(float)) );
    gpuErrCheck( cudaMalloc(&Q2_d, M*M*sizeof(float)) );
    gpuErrCheck( cudaMemcpy(R1_d, R, M*N*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrCheck( cudaMemcpy(Q1_d, Q, M*M*sizeof(float), cudaMemcpyHostToDevice) );

    printf("Malloc and memcpy done, calling Kernel...\n");

    int iter = 0;
    int swap = 0;
    while (1) {
        givens_gpu<<<blocks, threads>>>(
            Q1_d, R1_d, Q2_d, R2_d, M, N, leftmost, downmost, swap
        );
        gpuErrCheck( cudaPeekAtLastError() );
        gpuErrCheck( cudaDeviceSynchronize() );

        printf("Iteration %d done.\n", ++iter);

        size_t i = N - 1;
        for (;;) {
          size_t start = i ? downmost[i - 1] + 1 : 0;
          size_t end = downmost[i];
          for (size_t j = 1 + (start + end) / 2; j <= end; j++) leftmost[j]++;
          
          downmost[i] -= (1 + end - start) / 2;
          if (i == 0) break;
          else i--;
        }
        
        if (verbose) {
          for (int i = 0; i < N; i++) printf("%zu ", downmost[i]);
          printf("\n");
          for (int i = 0; i < M; i++) printf("%zu ", leftmost[i]);
          printf("\n");
        }

        size_t mn = min(M, N);
        if (leftmost[mn - 1] == mn - 1) break;
        swap = swap ^ 1;
    }

    printf("Givens done after %d iterations.\nSwap is %d\n", iter, swap);    

    gpuErrCheck( cudaMemcpy(R, swap ? R1_d : R2_d, M*N*sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrCheck( cudaMemcpy(Q, swap ? Q1_d : Q2_d, M*M*sizeof(float), cudaMemcpyDeviceToHost) );

    // givens_factorization(R, M, N, Q);

    printf("Transposing Q...\n");

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < i; j++) {
            float t = Q[i*M + j];
            Q[i*M + j] = Q[j*M + i];
            Q[j*M + i] = t;
        }
    }
    
    if (verbose) print_matrix(R, M, N);
    if (verbose) print_matrix(Q, M, M);

    if (check) {
        printf("Checking A...\n");

        float* reconstructed_A = (float*) malloc (sizeof(float) * M * N);
        matmul_f(Q, R, reconstructed_A, M, M, N);


        printf("Error: %.3f\n", max_err(A, reconstructed_A, M, N));
    }

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
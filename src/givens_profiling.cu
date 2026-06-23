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



float* solve_least_squares_Givens(float *A, float *b, int M, int N, int warmups,
    double* totalTime,
    double* givensTime, 
    double* updateLeftmostTime, 
    double* updateDownmostTime 
) {

    *totalTime = 0;
    *givensTime = 0;
    *updateDownmostTime = 0;
    *updateLeftmostTime = 0;

    float* Rb = (float*) malloc (sizeof(float) * M * (N + 1));
    for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
        Rb[i*(N + 1) + j] = A[i*N + j];
    for (int i = 0; i < M; i++)
        Rb[i*(N + 1) + N] = b[i];
    
    int threads = 256;
    int blocks = (M*(N + 1) + 255) / 256;

    int* leftmost = (int*) malloc (sizeof(int) * M);
    int* leftmost_d;
    int* downmost = (int*) malloc (sizeof(int) * N);
    int* downmost_d;

    gpuErrCheck( cudaMalloc(&leftmost_d, sizeof(int)*M) );
    gpuErrCheck( cudaMalloc(&downmost_d, sizeof(int)*N) );
    for (int i = 0; i < M; i++) leftmost[i] = 0;
    for (int i = 0; i < N; i++) downmost[i] = M - 1;
    gpuErrCheck( cudaMemcpy(leftmost_d, leftmost, M*sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrCheck( cudaMemcpy(downmost_d, downmost, N*sizeof(int), cudaMemcpyHostToDevice) );
    free(leftmost);
    free(downmost);

    float* Rb1_d;
    float* Rb2_d;
    gpuErrCheck( cudaMalloc(&Rb1_d, M*(N + 1)*sizeof(float)) );
    gpuErrCheck( cudaMalloc(&Rb2_d, M*(N + 1)*sizeof(float)) );
    gpuErrCheck( cudaMemcpy(Rb1_d, Rb, M*(N + 1)*sizeof(float), cudaMemcpyHostToDevice) );

    int iter = 0;
    int swap = 0;
    int mn = min(M, N);
    if (M <= N) mn--;
    int mxiters = (32 - __builtin_clz(M)) * N;
    int* last = (int*) malloc(sizeof(int));

    float milliseconds;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaStream_t leftmost_cpy_stream1;
    cudaStreamCreate(&leftmost_cpy_stream1);

    while (warmups--) givens_gpu_LLS<<<blocks, threads>>>(
        Rb1_d, Rb2_d,
        M, N, leftmost_d, downmost_d
    );

    struct timespec start_cpu, end_cpu;
    clock_gettime(CLOCK_MONOTONIC, &start_cpu);

    while (mxiters--) {
        
        // Givens Kernel
        cudaEventRecord(start);
        givens_gpu_LLS<<<blocks, threads>>>(
            Rb1_d, Rb2_d,
            M, N, leftmost_d, downmost_d
        );
        gpuErrCheck( cudaDeviceSynchronize() );

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&milliseconds, start, end);
        *givensTime += milliseconds;

        // update leftmost
        int blocksUpdLeft = (M + threads - 1) / threads;
        cudaEventRecord(start);
        update_leftmost<<<blocksUpdLeft, threads>>>(
            leftmost_d, downmost_d, M, N
        );
        gpuErrCheck( cudaDeviceSynchronize() );

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&milliseconds, start, end);
        *updateLeftmostTime += milliseconds;

        // update downmost
        cudaEventRecord(start);

        update_downmost<<<1, N>>>(downmost_d);
        cudaMemcpyAsync(last, leftmost_d + mn, sizeof(int), cudaMemcpyDeviceToHost, leftmost_cpy_stream1);
        
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&milliseconds, start, end);
        *updateDownmostTime += milliseconds;

        if (*last == mn) break;

        float* tmp = Rb1_d;
        Rb1_d = Rb2_d;
        Rb2_d = tmp;
    }
    cudaStreamDestroy(leftmost_cpy_stream1);
    free(last);

    clock_gettime(CLOCK_MONOTONIC, &end_cpu);

    *totalTime = (end_cpu.tv_sec - start_cpu.tv_sec)
                    + (end_cpu.tv_nsec - start_cpu.tv_nsec) * 1e-9;
    *totalTime *= 1000.0f;

    gpuErrCheck( cudaMemcpy(Rb, Rb2_d, M*(N + 1)*sizeof(float), cudaMemcpyDeviceToHost) );

    float* ans = (float*) malloc (sizeof(float) * N);
    for (int i = N - 1; i >= 0; i--) {
        float rhs = Rb[i*(N + 1) + N];
        for (int j = N - 1; j > i; j--) {
            rhs -= ans[j] * Rb[i*(N + 1) + j];
        }
        ans[i] = rhs / Rb[i*(N + 1) + i];
    }


    gpuErrCheck( cudaFree(Rb1_d) );    
    gpuErrCheck( cudaFree(Rb2_d) );
    gpuErrCheck( cudaFree(leftmost_d) );
    gpuErrCheck( cudaFree(downmost_d) );
    free(Rb);

    return ans;
}


int main(int argc, char* argv[]) {

    mt_seed(42);

    const int Ms[] = {1000, 100000, 1000000};
    const int Ns[] = {20, 40};

    const int numM = sizeof(Ms) / sizeof(Ms[0]);
    const int numN = sizeof(Ns) / sizeof(Ns[0]);

    int iters = 100;
    int warmups = 5;

    printf("\n");
    printf("=========================================================================================\n");
    printf("%-25s", "Metric");

    for (int m = 0; m < numM; m++) {
        for (int n = 0; n < numN; n++) {
            printf("| M=%-7d N=%-3d ", Ms[m], Ns[n]);
        }
    }
    printf("\n");
    printf("=========================================================================================\n");

    double totalTimes[3][2];
    double givensTimes[3][2];
    double leftmostTimes[3][2];
    double downmostTimes[3][2];

    for (int m = 0; m < numM; m++) {
        for (int n = 0; n < numN; n++) {

            int M = Ms[m];
            int N = Ns[n];

            float *A, *b;
            generate_random(&A, M, N);
            generate_random(&b, M, 1);

            double totalTotal = 0.0;
            double totalGivens = 0.0;
            double totalUpdateLeftmost = 0.0;
            double totalUpdateDownmost = 0.0;

            for (int iter = 0; iter < iters; iter++) {

                double totalTime;
                double givensTime;
                double leftmostTime;
                double downmostTime;

                float* v = solve_least_squares_Givens(
                    A, b, M, N, warmups,
                    &totalTime,
                    &givensTime,
                    &leftmostTime,
                    &downmostTime
                );

                totalTotal += totalTime;
                totalGivens += givensTime;
                totalUpdateLeftmost += downmostTime;
                totalUpdateDownmost += leftmostTime;
                
                free(v);
            }


            totalTotal /= iters;
            totalGivens /= iters;
            totalUpdateLeftmost /= iters;
            totalUpdateDownmost /= iters;

            totalTimes[m][n] = totalTotal;
            givensTimes[m][n] = totalGivens;
            leftmostTimes[m][n] = totalUpdateLeftmost; 
            downmostTimes[m][n] = totalUpdateDownmost; 

            free(A);
            free(b);
        }
    }

    printf("%-25s", "Total Time (ms)");
    for (int m = 0; m < numM; m++)
        for (int n = 0; n < numN; n++)
            printf("| %15.3f ", totalTimes[m][n]);
    printf("\n");

    printf("%-25s", "Givens Time (ms)");
    for (int m = 0; m < numM; m++)
        for (int n = 0; n < numN; n++)
            printf("| %15.3f ", givensTimes[m][n]);
    printf("\n");

    printf("%-25s", "Update Leftmost Time (ms)");
    for (int m = 0; m < numM; m++)
        for (int n = 0; n < numN; n++)
            printf("| %15.3f ", leftmostTimes[m][n]);
    printf("\n");

    printf("%-25s", "Update Downmost Time (ms)");
    for (int m = 0; m < numM; m++)
        for (int n = 0; n < numN; n++)
            printf("| %15.3f ", downmostTimes[m][n]);
    printf("\n");

    printf("%-25s", "Overhead Time (ms)");
    for (int m = 0; m < numM; m++)
        for (int n = 0; n < numN; n++)
            printf("| %15.3f ", totalTimes[m][n] - givensTimes[m][n] - leftmostTimes[m][n] - downmostTimes[m][n]);
    printf("\n");

    printf("==========================================================================================\n");

    return 0;
}
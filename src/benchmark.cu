#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "givens.h"
#include <time.h>
#include "matrix_ops.h"

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

#define gpuCuSolverCheck(call) do {                       \
    cusolverStatus_t status = call;                   \
    if (status != CUSOLVER_STATUS_SUCCESS) {          \
        printf("cuSOLVER error\n");                   \
        exit(1);                                      \
    }                                                 \
} while(0)


float compute_residual(float *A, float *x, float *b, int M, int N) {
    float *Ax = (float*)malloc(M * sizeof(float));
    matmul_f(A, x, Ax, M, N, 1);
    float err = 0.0f;
    for (int i = 0; i < M; i++) {
        float r = Ax[i] - b[i];
        err += r * r;
    }
    free(Ax);
    return err;
}

float* measure_givens(
    float *A, 
    float *b, 
    int M, int N, 
    double* kernelTime, double* totalTime,
    int warmup
) { 

    float* Rb = (float*) malloc (sizeof(float) * M * (N + 1));
    for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++)
        Rb[i*(N + 1) + j] = A[i*N + j];
    for (size_t i = 0; i < M; i++)
        Rb[i*(N + 1) + N] = b[i];
    
    int threads = 256;
    int blocks = (M*(N + 1) + 255) / 256;
    size_t* leftmost;
    size_t* downmost;
    cudaMallocManaged(&leftmost, sizeof(size_t)*M);
    cudaMallocManaged(&downmost, sizeof(size_t)*N);
    for (int i = 0; i < M; i++) leftmost[i] = 0;
    for (int i = 0; i < N; i++) downmost[i] = M - 1;

    float* Rb1_d;
    float* Rb2_d;
    gpuErrCheck( cudaMalloc(&Rb1_d, M*(N + 1)*sizeof(float)) );
    gpuErrCheck( cudaMalloc(&Rb2_d, M*(N + 1)*sizeof(float)) );
    gpuErrCheck( cudaMemcpy(Rb1_d, Rb, M*(N + 1)*sizeof(float), cudaMemcpyHostToDevice) );

    // malloc and memcpy is assumed to not be a part of the process.
    *kernelTime = 0;
    struct timespec start_cpu, end_cpu;
    clock_gettime(CLOCK_MONOTONIC, &start_cpu);

    while (warmup--) givens_gpu_LLS<<<blocks, threads>>>(
        Rb1_d, Rb2_d,
        M, N, leftmost, downmost, 0
    ); // repeatedly process and write to Rb2d, shouldnt be a problem 
    // since Rb2_d is assumed to be garbage values at the start.

    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);

    int iter = 0;
    int swap = 0;
    while (1) {
        cudaEventRecord(start_cuda);
        {
            givens_gpu_LLS<<<blocks, threads>>>(
                Rb1_d, Rb2_d,
                M, N, leftmost, downmost, swap
            );
            // gpuErrCheck( cudaDeviceSynchronize() );
            // gpuErrCheck( cudaPeekAtLastError() );
        }
        cudaEventRecord(stop_cuda);    
        cudaEventSynchronize(stop_cuda);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda);
        *kernelTime += milliseconds;
        // printf("Kernel iter %d: %f\n", iter + 1, milliseconds);

        size_t i = N - 1;
        for (;;) {
          size_t start = i ? downmost[i - 1] + 1 : 0;
          size_t end = downmost[i];
          for (size_t j = 1 + (start + end) / 2; j <= end; j++) leftmost[j]++;
          
          downmost[i] -= (1 + end - start) / 2;
          if (i == 0) break;
          else i--;
        }
        iter++;
        size_t mn = min(M, N);
        if (downmost[mn - 1] == mn - 1 && leftmost[mn - 1] == mn - 1) break;
        swap = swap ^ 1;

    }

    clock_gettime(CLOCK_MONOTONIC, &end_cpu);

    *totalTime = (end_cpu.tv_sec - start_cpu.tv_sec)
                    + (end_cpu.tv_nsec - start_cpu.tv_nsec) * 1e-9;
    *totalTime *= 1000.0f;

    gpuErrCheck( cudaMemcpy(Rb, swap ? Rb1_d : Rb2_d, M*(N + 1)*sizeof(float), cudaMemcpyDeviceToHost) );

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
    gpuErrCheck( cudaFree(leftmost) );
    gpuErrCheck( cudaFree(downmost) );
    free(Rb);

    return ans;
}


float* measure_cusolver(
    float *A, 
    float *b, 
    int M, int N, 
    double* kernelTime, double* totalTime,
    int warmup
) {
    float *d_A, *d_b, *d_tau, *d_work;
    int *d_info;
    int h_info;
    int lwork_geqrf = 0, lwork_ormqr = 0, lwork = 0;

    cusolverDnHandle_t handle;
    gpuCuSolverCheck(cusolverDnCreate(&handle));

    gpuErrCheck(cudaMalloc(&d_A,   M * N * sizeof(float)));
    gpuErrCheck(cudaMalloc(&d_b,   M * sizeof(float)));
    gpuErrCheck(cudaMalloc(&d_tau, N * sizeof(float)));
    gpuErrCheck(cudaMalloc(&d_info, sizeof(int)));

    gpuErrCheck(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemcpy(d_b, b, M * sizeof(float),     cudaMemcpyHostToDevice));

    // ----------------------------------------------------------------
    // Step 1: Workspace query then QR factorization  A = Q * R
    //         On exit, d_A holds the packed Householder reflectors
    //         (below diagonal) and R (upper triangle). d_tau holds
    //         the scalar factors of each elementary reflector.
    // ----------------------------------------------------------------
    gpuCuSolverCheck(cusolverDnSgeqrf_bufferSize(handle, M, N, d_A, M, &lwork_geqrf));
    gpuCuSolverCheck(cusolverDnSormqr_bufferSize(
        handle,
        CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        M, 1, N,
        d_A, M,
        d_tau,
        d_b, M,
        &lwork_ormqr
    ));

    lwork = (lwork_geqrf > lwork_ormqr) ? lwork_geqrf : lwork_ormqr;
    gpuErrCheck(cudaMalloc(&d_work, lwork * sizeof(float)));


    // ----------------------------------------------------------------
    // Step 1.5: WARMUP SECTION, dummy kernel launches using throwaway buffers 
    // so that d_A and d_b remain intact for the actual benchmark run.
    // ----------------------------------------------------------------
    float *d_A_warm, *d_b_warm;
    gpuErrCheck(cudaMalloc(&d_A_warm, M * N * sizeof(float)));
    gpuErrCheck(cudaMalloc(&d_b_warm, M * sizeof(float)));

    for (int w = 0; w < warmup; w++) {
        gpuErrCheck(cudaMemcpy(d_A_warm, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrCheck(cudaMemcpy(d_b_warm, b, M * sizeof(float),     cudaMemcpyHostToDevice));

        gpuCuSolverCheck(cusolverDnSgeqrf(handle, M, N, d_A_warm, M, d_tau, d_work, lwork, d_info));
        gpuCuSolverCheck(cusolverDnSormqr(
            handle,
            CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
            M, 1, N,
            d_A_warm, M,
            d_tau,
            d_b_warm, M,
            d_work, lwork,
            d_info
        ));
    }
    cudaDeviceSynchronize();

    cudaFree(d_A_warm);
    cudaFree(d_b_warm);

    // CPU TIMING START here.
    *kernelTime = 0;
    struct timespec start_cpu, end_cpu;
    clock_gettime(CLOCK_MONOTONIC, &start_cpu);
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    { // QR.
        cudaEventRecord(start_cuda);
        gpuCuSolverCheck(cusolverDnSgeqrf(handle, M, N, d_A, M, d_tau, d_work, lwork, d_info));
        cudaEventRecord(stop_cuda);
        cudaEventSynchronize(stop_cuda);
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda);
    *kernelTime += milliseconds;

    gpuErrCheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "sgeqrf failed: %d\n", h_info);
        exit(1);
    }

    // ----------------------------------------------------------------
    // Step 2: Apply Q^T to b in-place:  b <- Q^T * b
    //         After this, the first N entries of d_b are Q^T*b
    //         restricted to the column space; entries [N..M-1] are
    //         the residual components.
    // ----------------------------------------------------------------
    cudaEventRecord(start_cuda);
    gpuCuSolverCheck(cusolverDnSormqr(
        handle,
        CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        M, 1, N,
        d_A, M,
        d_tau,
        d_b, M,
        d_work, lwork,
        d_info
    ));
    cudaEventRecord(stop_cuda);
    cudaEventSynchronize(stop_cuda);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda);
    *kernelTime += milliseconds;

    gpuErrCheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "sormqr failed: %d\n", h_info);
        exit(1);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_cpu);
    *totalTime = (end_cpu.tv_sec - start_cpu.tv_sec)
                    + (end_cpu.tv_nsec - start_cpu.tv_nsec) * 1e-9;
    *totalTime *= 1000.0f;

    // ----------------------------------------------------------------
    // Step 3: Copy R and Q^T*b back to host for back substitution.
    //         R is the N×N upper triangle of the M×N packed matrix.
    // ----------------------------------------------------------------
    float* h_A = (float*)malloc(M * N * sizeof(float));
    float* h_b = (float*)malloc(M * sizeof(float));
    float* h_x = (float*)malloc(N * sizeof(float));

    gpuErrCheck(cudaMemcpy(h_A, d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrCheck(cudaMemcpy(h_b, d_b, M * sizeof(float),     cudaMemcpyDeviceToHost));

    // ----------------------------------------------------------------
    // Step 4: Manual back substitution  R * x = (Q^T b)[0:N]
    //         R is upper triangular, stored column-major with lda=M.
    // ----------------------------------------------------------------
    for (int i = N - 1; i >= 0; i--) {
        float sum = h_b[i];
        for (int j = i + 1; j < N; j++) {
            sum -= h_A[j * M + i] * h_x[j];  // column-major: R[i,j] = h_A[j*M + i]
        }
        h_x[i] = sum / h_A[i * M + i];        // R[i,i] = h_A[i*M + i]
    }

    free(h_A);
    free(h_b);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_tau);
    cudaFree(d_work);
    cudaFree(d_info);
    cusolverDnDestroy(handle);

    return h_x;
}

int main(int argc, char* argv[]) {

    int trials = 15;
    int warmup = 5;
    int test_count = 8;

    size_t MM[test_count] = {
        100,
        1000, 
        1000,
        10000,
        10000,
        100000,
        1000,
        3000000
    };

    size_t NN[test_count] = {
        10,
        32,
        100,
        32,
        64,
        32,
        1000,
        5,
    };

    printf("| %3s | %8s | %8s | %12s | %12s | %12s | %12s | \n",
          "ID", "M", "N", "Method", "Kernel Time", "Total Time", "Residual");
    printf("=========================================================================================\n");

    for (int i = 0; i < test_count; i++)
    {
        size_t M = MM[i], N = NN[i];

        double k_cusolver_sum = 0.0, t_cusolver_sum = 0.0, err_cusolver_sum = 0.0;
        double k_givens_sum   = 0.0, t_givens_sum   = 0.0, err_givens_sum   = 0.0;

        float *A, *b;
        generate_random(&A, M, N);
        generate_random(&b, M, 1);

        for (int t = 0; t < trials; t++)
        {

            double k_cusolver, t_cusolver;
            double k_givens, t_givens;

            float* v_cusolver = measure_cusolver(A, b, M, N, &k_cusolver, &t_cusolver, warmup);
            float* v_givens   = measure_givens(A, b, M, N, &k_givens, &t_givens, warmup);

            float err_cusolver = compute_residual(A, v_cusolver, b, M, N);
            float err_givens   = compute_residual(A, v_givens, b, M, N);

            k_cusolver_sum += k_cusolver;
            t_cusolver_sum += t_cusolver;
            err_cusolver_sum += err_cusolver;

            k_givens_sum += k_givens;
            t_givens_sum += t_givens;
            err_givens_sum += err_givens;

            free(v_cusolver);
            free(v_givens);
        }

        free(A);
        free(b);
        double k_cusolver_avg = k_cusolver_sum / trials;
        double t_cusolver_avg = t_cusolver_sum / trials;
        double err_cusolver_avg = err_cusolver_sum / trials;

        double k_givens_avg = k_givens_sum / trials;
        double t_givens_avg = t_givens_sum / trials;
        double err_givens_avg = err_givens_sum / trials;

        printf("| %3d | %8zu | %8zu | %12s | %12.6f | %12.6f | %.6e |\n",
            i, M, N, "cuSolver", k_cusolver_avg, t_cusolver_avg, err_cusolver_avg);
        printf("| %3s | %8s | %8s | %12s | %12.6f | %12.6f | %.6e |\n",
            "", "", "", "Givens", k_givens_avg, t_givens_avg, err_givens_avg);

        printf("-----------------------------------------------------------------------------------------\n");
    }
}
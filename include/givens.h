#include <math.h>
#include <assert.h>

void givens_calculate_cs(float a, float b, float* c, float* s) {
    float r = sqrt(a * a + b * b);
    *c = a / r;
    *s = -b / r;
}

void givens_rotation(float* A, int M, int N, int r1, int r2, int col, float* Q) {
    
    float c, s;
    givens_calculate_cs(A[r1*N + col], A[r2*N + col], &c, &s);
    for (int j = col; j < N; j++) {
        float a = c * A[r1*N + j] - s * A[r2*N + j];    
        float b = s * A[r1*N + j] + c * A[r2*N + j];    
        A[r1*N + j] = a;
        A[r2*N + j] = b;

    }

    for (int j = 0; j < M; j++) {
        float a = c * Q[r1*M + j] - s * Q[r2*M + j];    
        float b = s * Q[r1*M + j] + c * Q[r2*M + j];    
        Q[r1*M + j] = a;
        Q[r2*M + j] = b;
    }
}

/** 
* Assumes Q is already initialized to be I
*/
void givens_factorization(float* A, int M, int N, float* Q) {
    for (int j = 0; j < N; j++) {
        for (int i = M - 1; i > j; i--) {
            givens_rotation(A, M, N, i - 1, i, j, Q);
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < i; j++) {
            float t = Q[i*M+j];
            Q[i*M+j] = Q[j*M+i];
            Q[j*M+i] = t;
        }
    }
}







__global__ void givens_gpu(
    float* Q1, float* R1, 
    float* Q2, float* R2,
    size_t M, 
    size_t N, 
    size_t* leftmost, 
    size_t* downmost,
    int is_swap
) {
    float* Qsrc = is_swap ? Q2 : Q1;
    float* Qdst = is_swap ? Q1 : Q2;
    float* Rsrc = is_swap ? R2 : R1;
    float* Rdst = is_swap ? R1 : R2;

    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx >= M*M + M*N) return;

    char i_am_q = tidx < M * M;

    size_t i = i_am_q ? tidx / M : (tidx - M*M) / N;
    size_t j = i_am_q ? tidx % M : (tidx - M*M) % N;

    {
        // this column is the source of rotation for this cell
        size_t col = leftmost[i];

        // if (!i_am_q)
        // printf("leftmost[%lld] is %lld\n", i, col);

        // the "region of work" for this cell's column
        size_t start = col ? downmost[col - 1] + 1 : 0;
        size_t end = col < N ? downmost[col] : start; // doesnt really matter 
        size_t length = 1 + (end - start);
        char is_lower_half = i >= (start + (length + 1) / 2);
        float a = is_lower_half ? Rsrc[(i - length / 2)*N + col] : Rsrc[i*N + col];
        float b = is_lower_half ? Rsrc[i*N + col] : Rsrc[(i + length / 2)*N + col];
        float r = sqrt(a * a + b * b);
        float c = a / r;
        float s = -b / r;
        float res;

        float* ptr = i_am_q ? Qsrc : Rsrc;
        size_t stride = i_am_q ? M : N;
        float prev_val = ptr[i*stride + j];
        float r1 = is_lower_half ? ptr[(i - length / 2)*stride + j] : ptr[i*stride + j];
        float r2 = is_lower_half ? ptr[i*stride + j] : ptr[(i + length / 2)*stride + j];
        int is_do_work = (length > 1) && !((length % 2 == 1) && start == i);
        res = is_do_work ? ( is_lower_half ? s * r1 + c * r2 : c * r1 - s * r2 ) : prev_val;

        ptr = i_am_q ? Qdst : Rdst;
        ptr[i*stride + j] = res;

        // if (is_do_work && !i_am_q) {
        //     // printf("Thread %d is do work, i=%lld, j=%lld.\n", tidx, i, j);
        //     if (col == j && end == i) { downmost[col] = downmost[col] - length / 2; }
        //     if (col == j && is_lower_half) { leftmost[i]++; }
        // }
    }
}



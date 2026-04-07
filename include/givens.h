#include <math.h>

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
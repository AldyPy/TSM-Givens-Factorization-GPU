#include <math.h>
#include <stdio.h>
#include "ta_colors.h"

int read_matrix(const char* filename, int* M, int* N, float** A) {
    FILE* f = fopen(filename, "r");
    if (!f) return -1;

    // read dimensions
    if (fscanf(f, "%d %d", M, N) != 2) {
        fclose(f);
        return -2;
    }

    int size = (*M) * (*N);
    *A = (float*)malloc(sizeof(float) * size);
    if (!*A) {
        fclose(f);
        return -3;
    }

    // read data (row-major)
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%f", &((*A)[i])) != 1) {
            free(*A);
            fclose(f);
            return -4;
        }
    }

    fclose(f);
    return 0;
}

void print_matrix(const float* A, int M, int N) {
    printf("%sMatrix %dx%d.%s\n", TA_FG_WHITE, M, N, TA_COLOR_RESET);
    for (int i = 0; i < M; i++) {
        printf("%s[ %s", TA_FG_WHITE, TA_COLOR_RESET);
        for (int j = 0; j < N; j++) {
            float val = A[i * N + j];
            // Determine color based on absolute value
            const char* color;
            float abs_val = fabsf(val);
            if (abs_val < 0.1f) {
                color = TA_FG_GRAY;      // Very small values
            } else if (abs_val < 0.5f) {
                color = TA_FG_GREEN;     // Small values
            } else if (abs_val < 1.0f) {
                color = TA_FG_YELLOW;    // Medium values
            } else if (abs_val < 5.0f) {
                color = TA_FG_BLUE;      // Larger values
            } else if (abs_val < 10.0f) {
                color = TA_FG_MAGENTA;   // Even larger
            } else if (abs_val < 50.0f) {
                color = TA_FG_CYAN;      // Very large
            } else {
                color = TA_FG_RED;       // Extremely large
            }
            printf("%s%8.6f%s", color, val, TA_COLOR_RESET);
            if (j != M - 1) printf(", ");
        }
        printf("%s ]\n%s", TA_FG_WHITE, TA_COLOR_RESET);
    }
}

void matmul_f(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++) {
            C[i*K + k] = 0;
            for (int j = 0; j < N; j++) {
                C[i*K + k] = fmaf(A[i*N + j], B[j*K + k], C[i*K + k]);
            }
        }
    }
}

float max_err(const float* A, const float* B, int M, int N) {
    float res = 0;
    for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
    res = max(res, abs(A[i*N+j] - B[i*N+j]));
    return res;
}
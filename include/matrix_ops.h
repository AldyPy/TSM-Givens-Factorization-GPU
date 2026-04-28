#include <math.h>
#include <stdio.h>
#include "ta_colors.h"

int read_matrix(const char* filename, size_t* M, size_t* N, float** A) {
    FILE* f = fopen(filename, "r");
    if (!f) return -1;

    // read dimensions
    if (fscanf(f, "%zu %zu", M, N) != 2) {
        fclose(f);
        return -2;
    }

    size_t size = (*M) * (*N);
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


void print_val(const char* color, double val) {
    // (1) clamp near-zero
    if (fabs(val) < 1e-9) val = 0.0;

    // count digits before decimal
    double absv = fabs(val);
    int digits_before = (absv < 1.0) ? 1 : (int)floor(log10(absv)) + 1;

    // (2) total width = 8 → precision = remaining
    int precision = 8 - digits_before;
    if (precision < 0) precision = 0;

    printf("%s%s%.*f%s",
           color,
           val < 0 ? "" : " ",
           precision,
           val,
           TA_COLOR_RESET);
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
            print_val(color, val);
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


#include <stdint.h>
#include <time.h>

#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfU
#define MT_UPPER_MASK 0x80000000U
#define MT_LOWER_MASK 0x7fffffffU

static uint32_t mt[MT_N];
static int mt_index = MT_N + 1;

/* Initialize with seed */
void mt_seed(uint32_t seed) {
    mt[0] = seed;
    for (int i = 1; i < MT_N; i++) {
        mt[i] = 1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
    }
    mt_index = MT_N;
}

/* Generate next state */
static void mt_twist(void) {
    for (int i = 0; i < MT_N; i++) {
        uint32_t y = (mt[i] & MT_UPPER_MASK) | (mt[(i + 1) % MT_N] & MT_LOWER_MASK);
        mt[i] = mt[(i + MT_M) % MT_N] ^ (y >> 1);
        if (y & 1)
            mt[i] ^= MT_MATRIX_A;
    }
    mt_index = 0;
}

/* Extract 32-bit random int */
uint32_t mt_rand_u32(void) {
    if (mt_index >= MT_N)
        mt_twist();

    uint32_t y = mt[mt_index++];

    /* Tempering */
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= y >> 18;

    return y;
}

/* Uniform float in [0,1) */
float mt_rand_float(void) {
    return (mt_rand_u32() >> 8) * (1.0f / 16777216.0f);
}

/* Example init */
void rng_init(void) {
    mt_seed((uint32_t)time(NULL));
}


void generate_random(float **A, int M, int N) {
    *A = (float*)malloc(M * N * sizeof(float));
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            (*A)[i + j*M] = mt_rand_float() * 100.f;
        }
    }
}


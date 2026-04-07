#include <math.h>

void givens_calculate_cs(float a, float b, float* c, float* s) {
    r = sqrt(a * a + b * b);
    *c = a / r;
    *s = -b / r;
}

void givens_rotation(float* A, int M, int N, int r1, int r2, int col) {
    
    float c, float s;
    
    for (int j = col; j < N; j++) {

    }
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s M N\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    printf("%d %d\n", M, N);

    srand((unsigned)time(NULL));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float x = (float)rand() / RAND_MAX;  // [0,1]
            printf("%.6f ", x);
        }
        printf("\n");
    }

    return 0;
}
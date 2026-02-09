#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <immintrin.h> /* The header for all SIMD intrinsics */

#include "poly_mult_avx2.h"

#define BI 256   /* block size for A (tune) */
#define BJ 256   /* block size for B (tune) */

void poly_mult_avx2(const int * restrict poly_a_in, size_t deg_a, const int * restrict poly_b_in, size_t deg_b, int * restrict poly_res_out, double * time_out) {
    struct timespec start, finish;
    double time_spent;
    /* Initialize output buffer to 0 */
    memset(poly_res_out, 0, ROUND_UP_8(deg_a + deg_b + 1 + 8) * sizeof(int));

    /* Main compute loop */
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    for (size_t i_block = 0; i_block <= deg_a; i_block += 2) {
        size_t i_end = (i_block + 2 <= deg_a + 1) ? i_block + 2 : deg_a + 1;

        for (size_t j_block = 0; j_block <= deg_b; j_block += 8) {
            // size_t j_end = (j_block + 8 <= deg_b + 1) ? j_block + 8 : deg_b + 1;

            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();

            for (size_t i = i_block; i < i_end; i++) {
                __m256i a_vec = _mm256_set1_epi32(poly_a_in[i]);

                __m256i b_vec = _mm256_loadu_si256((__m256i*)&poly_b_in[j_block]);

                __m256i prod = _mm256_mullo_epi32(a_vec, b_vec);

                if (i == i_block)
                    acc0 = _mm256_add_epi32(acc0, prod);
                else
                    acc1 = _mm256_add_epi32(acc1, prod);
            }

            // write back once
            _mm256_storeu_si256((__m256i*)&poly_res_out[i_block + j_block], acc0);

            if (i_block + 1 < i_end)
                _mm256_storeu_si256((__m256i*)&poly_res_out[i_block + 1 + j_block], acc1);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    time_spent = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1e9; 

    if (time_out != NULL) *time_out = time_spent;

    return;
}
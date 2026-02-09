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
    for (size_t i_block = 0; i_block <= deg_a; i_block += BI) {
        size_t i_end = (i_block + BI <= deg_a + 1) ? i_block + BI : deg_a + 1;

        for (size_t j_block = 0; j_block <= deg_b; j_block += BJ) {
            size_t j_end = (j_block + BJ <= deg_b + 1) ? j_block + BJ : deg_b + 1;

            /* reuse this block of B for many A's */
            for (size_t i = i_block; i < i_end; i++) {
                __m256i a_vec = _mm256_set1_epi32(poly_a_in[i]);
                int *res_ptr = poly_res_out + i;

                size_t j = j_block;
                for (; j + 7 < j_end; j += 8) {
                    __m256i b_vec = _mm256_load_si256((__m256i*)&poly_b_in[j]);
                    __m256i c_vec = _mm256_load_si256((__m256i*)&res_ptr[j]);

                    __m256i prod = _mm256_mullo_epi32(a_vec, b_vec);
                    c_vec = _mm256_add_epi32(c_vec, prod);

                    _mm256_store_si256((__m256i*)&res_ptr[j], c_vec);
                }
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    time_spent = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1e9; 

    if (time_out != NULL) *time_out = time_spent;

    return;
}
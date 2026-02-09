#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <immintrin.h> /* The header for all SIMD intrinsics */

#include "poly_add_avx2.h"

void poly_add_avx2(const int * restrict poly_a_in, const int * restrict poly_b_in, int * restrict poly_res_out, size_t size, double * time_out) {
    struct timespec start, finish;
    double time_spent;

    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    /* Vectorized loop: process 8 integers per iteration (AVX2 256-bit registers) */
    for (size_t i = 0; i < size; i += 8) {
        /* Load 8 integers from poly_a_in and poly_b_in (aligned load) */
        __m256i a = _mm256_load_si256((__m256i*)&poly_a_in[i]);
        __m256i b = _mm256_load_si256((__m256i*)&poly_b_in[i]);

        /* Sum the two input vectors */
        __m256i sum = _mm256_add_epi32(a, b);

        /* Store the final result back into poly_res_out (aligned store) */
        _mm256_store_si256((__m256i*)&poly_res_out[i], sum);
    }

    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    time_spent = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1e9; 

    if (time_out != NULL) *time_out = time_spent;

    return;
}
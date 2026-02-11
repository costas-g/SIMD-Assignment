#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <immintrin.h> /* The header for all SIMD intrinsics */

#include "poly_mult_avx2.h"
#include "util.h"

/* Only define one of these */
// #define UNTILED     // no tiled: use simple untiled inner and outer loops
// #define S_TILED     // single-tiled: tile only inner loop in blocks
#define D_TILED     // double-tiled: tile both inner and outer loops in blocks

#define BLOCK_I 4096 /* block size for A (tune) */
#define BLOCK_J 4096 /* block size for B (tune) */

void poly_mult_avx2(const int * restrict poly_a_in, size_t deg_a, const int * restrict poly_b_in, size_t deg_b, int * restrict poly_res_out, double * time_out) {
    struct timespec start, finish;
    double time_spent;

    /* array sizes */
    size_t deg_res = deg_a + deg_b;
    size_t size_a   = ROUND_UP_8(deg_a   + 1    ); /* number of elements of poly_a */
    size_t size_b   = ROUND_UP_8(deg_b   + 1    ); /* number of elements of poly_b */
    size_t size_res = ROUND_UP_8(deg_res + 1 + 8); /* number of elements of poly_res */

    /* Initialize output buffer to 0 */
    memset(poly_res_out, 0, size_res * sizeof(int));

    /* Main compute loop */
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */

    #ifdef UNTILED
    for (size_t i = 0; i < size_a; i++) {
        const int poly_ai = poly_a_in[i];       /* scalar value */
        int * poly_res_ptr = poly_res_out + i;  /* output offset */

        /* broadcast scalar to AVX2 register */
        __m256i a_vec = _mm256_set1_epi32(poly_ai);

        /* process 8 elements at a time */
        for(size_t j = 0; j < size_b; j += 8) {
            /* load 8 elements from poly_b_in */
            __m256i b_vec = _mm256_load_si256((__m256i*)&poly_b_in[j]);

            /* multiply 8 ints */
            __m256i prod_vec = _mm256_mullo_epi32(a_vec, b_vec);

            /* load current result */
            __m256i c_vec = _mm256_loadu_si256((__m256i*)&poly_res_ptr[j]);

            /* accumulate */
            c_vec = _mm256_add_epi32(c_vec, prod_vec);

            /* store back */
            _mm256_storeu_si256((__m256i*)&poly_res_ptr[j], c_vec);
        }
    }
    #endif

    #ifdef S_TILED
    
    #endif

    #ifdef D_TILED
    // Outer Loop 1: Tile 'i' (Control access to A and Res)
    for (size_t ii = 0; ii < size_a; ii += BLOCK_I) {
        size_t i_end = (ii + BLOCK_I > size_a) ? size_a : ii + BLOCK_I;

        // Outer Loop 2: Tile 'j' (Control access to B)
        for (size_t jj = 0; jj < size_b; jj += BLOCK_J) {
            size_t j_end = (jj + BLOCK_J > size_b) ? size_b : jj + BLOCK_J;

            
            for (size_t i = ii; i < i_end; i++) {
                const int poly_ai = poly_a_in[i];       /* scalar value */
                int * poly_res_ptr = poly_res_out + i;  /* output offset */

                /* broadcast scalar to AVX2 register */
                __m256i a_vec = _mm256_set1_epi32(poly_ai);

                /* process 8 elements at a time */
                for(size_t j = jj; j < j_end; j += 8) {
                    /* load 8 elements from poly_b_in */
                    __m256i b_vec = _mm256_load_si256((__m256i*)&poly_b_in[j]);

                    /* multiply 8 ints */
                    __m256i prod_vec = _mm256_mullo_epi32(a_vec, b_vec);

                    /* load current result */
                    __m256i c_vec = _mm256_loadu_si256((__m256i*)&poly_res_ptr[j]);

                    /* accumulate */
                    c_vec = _mm256_add_epi32(c_vec, prod_vec);

                    /* store back */
                    _mm256_storeu_si256((__m256i*)&poly_res_ptr[j], c_vec);
                }
            }
        }
    }
    #endif

    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    time_spent = time_delta(&start, &finish);

    if (time_out != NULL) *time_out = time_spent;

    return;
}
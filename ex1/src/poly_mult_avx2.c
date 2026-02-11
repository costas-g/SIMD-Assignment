#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <immintrin.h> /* The header for all SIMD intrinsics */

#include "poly_mult_avx2.h"
#include "util.h"

/* Only define one of these */
// #define UNTILED     // no tiled: use simple untiled inner and outer loops
#define S_TILED     // single-tiled: tile only inner loop in blocks
// #define D_TILED     // double-tiled: tile both inner and outer loops in blocks

#define BLOCK_I 512 /* block size for A (tune) */
#define BLOCK_J 512 /* block size for B (tune) */

void poly_mult_avx2(const int * restrict poly_a_in, size_t deg_a, const int * restrict poly_b_in, size_t deg_b, int * restrict poly_res_out, double * time_out) {
    struct timespec start, finish;
    double time_spent;

    #ifdef DEBUG
    int tmp[8];
    #endif

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

    #ifdef D_TILED
    // Outer Loop 1: Tile 'i' (Control access to A and Res)
    for (size_t ii = 0; ii < size_a; ii += BLOCK_I) {
        // size_t i_end = (ii + BLOCK_I > size_a) ? size_a : ii + BLOCK_I;

        // Outer Loop 2: Tile 'j' (Control access to B)
        for (size_t jj = 0; jj < size_b; jj += BLOCK_J) {
            // size_t j_end = (jj + BLOCK_J > size_b) ? size_b : jj + BLOCK_J;

            
            /* if (jj & 7) == 0, i.e. jj multiple of 8, then load aligned */
            // Load 8 elements of b - but we only use 4 each time
            __m256i b_vec = _mm256_loadu_si256((__m256i*)(poly_b_in + jj));
            /* else [if (jj & 7) != 0 and (jj & 3) == 0], i.e. jj multiple of 4 but not of 8, then swap halves */
            /* for the next jj+1 block of b, just swap halves. Save a load. */
            // b_vec = _mm256_permute2x128_si256(b_vec, b_vec, 0x01); // swap lower and upper halfs


            // Prepare accumulators
            __m256i acc0 = _mm256_setzero_si256();
            __m256i prev_res; // will be loaded from memory


            // --- Core Computation (Fits entirely in L1) ---

            // Process 4 elements of a
            for (size_t i = 0; i < BLOCK_I; i++) {
                // Broadcast a[ii + i] to all 8 lanes
                __m256i a_broadcast = _mm256_set1_epi32(poly_a_in[ii + i]);

                // Multiply
                __m256i prod = _mm256_mullo_epi32(a_broadcast, b_vec);

                // Discard upper half and shift accordingly
                __m256i prod_shifted = _mm256_permute2x128_si256(prod, prod, 0x80); // zero upper half;
                if (i > 0) {
                    // Shift left by i ints across the whole ymm using permutes
                    __m256i shifted = prod_shifted;//_mm256_permute2x128_si256(prod, prod, 0x08); // zero upper half
                    // shifted = _mm256_slli_si256(shifted, i * 4);                   // shift left within lower half
                    switch (i) { // need to unroll
                    case 0:
                        shifted = _mm256_slli_si256(prod, 0);
                        break;
                    case 1:
                        shifted = _mm256_slli_si256(prod, 4);
                        break;
                    case 2:
                        shifted = _mm256_slli_si256(prod, 8);
                        break;
                    case 3:
                        shifted = _mm256_slli_si256(prod, 12);
                        break;
                    }
                    // __m128i upper_from_lower = _mm_srli_si128(_mm256_castsi256_si128(prod), (4 - i) * 4);
                    __m128i upper_from_lower;
                    switch (i) { // need to unroll
                    case 0:
                        // shift by (4-0)*4 = 16 bytes → entire register becomes zero
                        upper_from_lower = _mm_srli_si128(_mm256_castsi256_si128(prod), 16);
                        break;
                    case 1:
                        // shift by (4-1)*4 = 12 bytes
                        upper_from_lower = _mm_srli_si128(_mm256_castsi256_si128(prod), 12);
                        break;
                    case 2:
                        // shift by (4-2)*4 = 8 bytes
                        upper_from_lower = _mm_srli_si128(_mm256_castsi256_si128(prod), 8);
                        break;
                    case 3:
                        // shift by (4-3)*4 = 4 bytes
                        upper_from_lower = _mm_srli_si128(_mm256_castsi256_si128(prod), 4);
                        break;
                    }
                    shifted = _mm256_inserti128_si256(shifted, upper_from_lower, 1); // fill upper half
                    prod_shifted = shifted;
                }

                // // Shift prod left by i ints (4 bytes each) within 128-bit halves
                // // __m256i prod_shift = _mm256_slli_si256(prod, i * 4);
                
                // // Handle crossing 128-bit boundary
                // if (i > 0) {
                //     // extract lower 128-bit half
                //     __m128i lower = _mm256_castsi256_si128(prod);
                //     // extract the int that crosses boundary
                //     // __m128i carry = _mm_srli_si128(lower, (4 - i) * 4);  // (4-i) ints to get correct element
                //     __m128i carry;
                //     switch (i) {
                //     case 0:
                //         // shift by (4-0)*4 = 16 bytes → entire register becomes zero
                //         carry = _mm_srli_si128(lower, 16);
                //         break;
                //     case 1:
                //         // shift by (4-1)*4 = 12 bytes
                //         carry = _mm_srli_si128(lower, 12);
                //         break;
                //     case 2:
                //         // shift by (4-2)*4 = 8 bytes
                //         carry = _mm_srli_si128(lower, 8);
                //         break;
                //     case 3:
                //         // shift by (4-3)*4 = 4 bytes
                //         carry = _mm_srli_si128(lower, 4);
                //         break;
                //     }
                //     // insert into upper half
                //     prod_shift = _mm256_inserti128_si256(prod_shift, carry, 1);
                // }
                
                // DEBUG: print prod_shifted
                #ifdef DEBUG
                _mm256_storeu_si256((__m256i*)tmp, prod_shifted);
                print_arr(tmp, 8); 
                #endif

                // Accumulate
                acc0 = _mm256_add_epi32(acc0, prod_shifted);
            }

            /* When ii+jj is a multiple of 8 (50% of the time) then the loads and stores are 32-byte aligned */
            /* when ((ii+jj) & 7) == 0 */

            // Load current result from memory
            prev_res = _mm256_loadu_si256((__m256i*)(poly_res_out + ii + jj));
            // Add the new partial result
            acc0 = _mm256_add_epi32(acc0, prev_res);
            // Write back result
            _mm256_storeu_si256((__m256i*)(poly_res_out + ii + jj), acc0);
        }
    }
    #endif

    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    time_spent = time_delta(&start, &finish);

    if (time_out != NULL) *time_out = time_spent;

    return;
}
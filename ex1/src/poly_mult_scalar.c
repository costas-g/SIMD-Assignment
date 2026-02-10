#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "poly_mult_scalar.h"
#include "util.h"

const size_t BLOCK_I = 512; 
const size_t BLOCK_J = 512;

void poly_mult_scalar(const int * restrict poly_a_in, size_t deg_a, const int * restrict poly_b_in, size_t deg_b, int * restrict poly_res_out, double * time_out) {
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
    // for (size_t i = 0; i < size_a; i++) {
    //     const int poly_ai = poly_a_in[i];       /* load poly_a_in values once per inner loop       */
    //     int * poly_res_ptr = poly_res_out + i;  /* compute result offset index once per inner loop */

    //     for(size_t j = 0; j < size_b; j++) {
    //         poly_res_ptr[j] += poly_ai * poly_b_in[j]; /* multiply */
    //     }
    // }
    // for (size_t jj = 0; jj < size_b; jj += BLOCK_SIZE) {
    //     size_t j_limit = (jj + BLOCK_SIZE > size_b) ? size_b : jj + BLOCK_SIZE;

    //     for (size_t i = 0; i < size_a; i++) {
    //         const int poly_ai = poly_a_in[i];               /* load poly_a_in values once per inner loop       */
    //         int * restrict poly_res_ptr = poly_res_out + i; /* compute result offset index once per inner loop */

    //         // This inner loop now works on a "cache-friendly" chunk
    //         for (size_t j = jj; j < j_limit; j++) {
    //             poly_res_ptr[j] += poly_ai * poly_b_in[j];  /* multiply */
    //         }
    //     }
    // }
    // Outer Loop 1: Tile 'i' (Control access to A and Res)
    for (size_t ii = 0; ii < size_a; ii += BLOCK_I) {
        size_t i_end = (ii + BLOCK_I > size_a) ? size_a : ii + BLOCK_I;

        // Outer Loop 2: Tile 'j' (Control access to B)
        for (size_t jj = 0; jj < size_b; jj += BLOCK_J) {
            size_t j_end = (jj + BLOCK_J > size_b) ? size_b : jj + BLOCK_J;

            // --- Core Computation (Fits entirely in L1) ---
            for (size_t i = ii; i < i_end; i++) {
                const int poly_ai = poly_a_in[i];               // Loaded from L1
                int * restrict poly_res_ptr = poly_res_out + i; // Offset calculated once

                // Inner loop vectorizes easily
                for (size_t j = jj; j < j_end; j++) {
                    poly_res_ptr[j] += poly_ai * poly_b_in[j]; // poly_b_in[j] in L1, poly_res_out[i+j] in L1
                }
            }
            // ----------------------------------------------
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    time_spent = time_diff(&start, &finish);

    if (time_out != NULL) *time_out = time_spent;

    return;
}
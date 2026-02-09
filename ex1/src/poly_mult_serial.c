#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "poly_mult_serial.h"

void poly_mult_serial(const int * restrict poly_a_in, size_t deg_a, const int * restrict poly_b_in, size_t deg_b, int * restrict poly_res_out, double * time_out) {
    struct timespec start, finish;
    double time_spent;

    /* Initialize output buffer to 0 */
    memset(poly_res_out, 0, (deg_a + deg_b + 1) * sizeof(int));

    /* Main compute loop */
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    for (size_t i = 0; i <= deg_a; i++) {
        const int poly_ai = poly_a_in[i];       /* load poly_a_in values once per inner loop       */
        int * poly_res_ptr = poly_res_out + i;  /* compute result offset index once per inner loop */

        for(size_t j = 0; j <= deg_b; j++) {
            poly_res_ptr[j] += poly_ai * poly_b_in[j]; /* multiply */
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    time_spent = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1e9; 

    if (time_out != NULL) *time_out = time_spent;

    return;
}
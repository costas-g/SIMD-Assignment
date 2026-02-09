#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "poly_add_serial.h"

void poly_add_serial(const int * restrict poly_a_in, const int * restrict poly_b_in, int * restrict poly_res_out, size_t size, double * time_out) {
    struct timespec start, finish;
    double time_spent;

    /* Main compute loop */
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    for (size_t i = 0; i < size; i++) {
        poly_res_out[i] = poly_a_in[i] + poly_b_in[i]; /* add */
    }
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    time_spent = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1e9; 

    if (time_out != NULL) *time_out = time_spent;

    return;
}
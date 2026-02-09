#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "poly_mult_serial.h"

void poly_mult_serial(const int * const poly_a_in, size_t deg_a, const int * const poly_b_in, size_t deg_b, int * const poly_res_out, double * const time_out) {
    struct timespec start, finish;
    double time_spent;

    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    for (size_t i = 0; i <= deg_a; i++){
        for(size_t j = 0; j <= deg_b; j++){
            poly_res_out[i+j] += poly_a_in[i] * poly_b_in[j];
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    time_spent = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1e9; 

    if (time_out != NULL) *time_out = time_spent;

    return;
}
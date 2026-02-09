#ifndef poly_util_h
#define poly_util_h

#include <stdio.h>

/* Counts the number of errors in matching two polynomials' coefficients.*/
static int poly_count_errors(const int * const poly_a, const int * const poly_b, size_t deg_common) {
    size_t num_errors = 0;
    size_t i;
    for (i = 0; i < deg_common + 1; i++) {
        if (poly_a[i] != poly_b[i]) {
            num_errors++;
        }
    }
    return num_errors;
}

#ifdef DEBUG
static void print_poly(const int * const poly, size_t deg) {
    size_t i;
    for (i = 0; i < deg + 1; i++) {
        printf("%d, ", poly[i]);
    }
    puts("");
}
#endif

#endif
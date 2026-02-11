#include <stdlib.h>

#define POS_COEFFS

void poly_random_fill(int * const poly_out, const size_t degree, int max_coeff) {
    /* compute final max coefficient */
    if (max_coeff < 1 || max_coeff > RAND_MAX/2) max_coeff = RAND_MAX;
    #ifndef POS_COEFFS
    else max_coeff = 2*max_coeff;
    #endif

    /* Random fill with non zeroes */
    for (size_t i = 0; i <= degree; i++){
        int coeff;

        #ifdef POS_COEFFS
        coeff = rand() % max_coeff + 1; /* in range [1, max_coeff] */
        #else
        do {
            coeff = rand() % max_coeff - max_coeff/2;   /* in range [-max_coeff, max_coeff-1] */
        }
        while (coeff == 0); /* coefficients should be non-zero integers */
        #endif

        poly_out[i] = (int) coeff; /* assign value */
    }

    return;
}
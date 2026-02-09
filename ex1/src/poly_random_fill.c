#include <stdio.h>
#include <stdlib.h>

void poly_random_fill(int * const poly_out, const size_t degree, int max_coeff) {
    /* compute final max coefficient */
    if (max_coeff < 1 || max_coeff > RAND_MAX/2) max_coeff = RAND_MAX;
    else max_coeff = 2*max_coeff;

    /* Random fill with non zeroes */
    for (size_t i = 0; i <= degree; i++){
        int coeff;
        do {
            coeff = rand() % max_coeff - max_coeff/2;   /* in range [-max_coeff, max_coeff-1] */
        }
        while (coeff == 0); /* coefficients should be non-zero integers */

        poly_out[i] = (int) coeff; /* assign value */
    }

    return;
}
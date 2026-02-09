#ifndef _poly_random_fill_h_
#define _poly_random_fill_h_

#include <stddef.h> /* defines size_t */

/* Random fill of the coeeficients of given polynomial with non-zero integers.*/
void poly_random_fill(int * const poly_out, const size_t degree, int max_coeff);

#endif

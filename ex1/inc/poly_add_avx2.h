#ifndef _poly_add_avx2_h_
#define _poly_add_avx2_h_

#include <stddef.h> /* defines size_t */

/* Adds two polynomials and saves the result to output buffer. Using AVX2. */
void poly_add_avx2(const int * restrict poly_a_in, const int * restrict poly_b_in, int * restrict poly_res_out, size_t size, double * time_out);

#endif
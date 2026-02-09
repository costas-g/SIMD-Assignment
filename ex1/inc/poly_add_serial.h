#ifndef _poly_add_serial_h_
#define _poly_add_serial_h_

#include <stddef.h> /* defines size_t */

/* Adds two polynomials and saves the result to output buffer. */
void poly_add_serial(const int * restrict poly_a_in, const int * restrict poly_b_in, int * restrict poly_res_out, size_t size, double * time_out);

#endif
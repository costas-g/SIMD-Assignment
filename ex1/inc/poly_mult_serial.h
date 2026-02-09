#ifndef _poly_mult_serial_h_
#define _poly_mult_serial_h_

#include <stddef.h> /* defines size_t */

/* Multiplies two polynomials and saves the result to output buffer. 
 * Output buffer must be at least `deg_a`+`deg_b`+`1` in size.
*/
void poly_mult_serial(const int * restrict poly_a_in, size_t deg_a, const int * restrict poly_b_in, size_t deg_b, int * restrict poly_res_out, double * time_out);

#endif
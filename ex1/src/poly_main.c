#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> /* Required for strerror */
#include <errno.h>  /* Required for errno */

#include "util.h"
#include "poly_util.h"
#include "poly_random_fill.h"
#include "poly_mult_scalar.h"
#include "poly_mult_avx2.h"

#define MAX_COEFF 1

void Usage(char* prog_name, const long long * const degree_input);

int main(int argc, char* argv[]) {
    size_t deg_a;   /* Input A polynomial degree */
    size_t deg_b;   /* Input B polynomial degree */
    size_t deg_res; /* Result polynomial degree  */

    /* ============================= Input ============================= */
    /* Parse inputs and error check */
    long long deg_input = 0;
    if (argc < 2) 
        Usage(argv[0], NULL);
    else {
        deg_input = strtoll(argv[1], NULL, 10); 
        if (deg_input < 1) Usage(argv[0], &deg_input); /* invalid degree value */
    }
    deg_a = (size_t) deg_input; /* polynomials have the same degree */
    deg_b = (size_t) deg_input; /* polynomials have the same degree */
    deg_res = deg_a + deg_b;    /* resultant polynomial's degree is the sum of the input polynomials' degrees */
        

    /* ========================= Common variables ========================= */
    /* Timing variables */
    struct timespec start, finish;
    // double elapsed_time;
    double gen_time;
    double scalar_time = 0.;
    double simd_time   = 0.;

    /* polynomial pointers */
    int *poly_a          = NULL;    /* input polynomial */
    int *poly_b          = NULL;    /* input polynomial */
    int *poly_res_scalar = NULL;    /* resultant polynomial from scalar mult */
    int *poly_res_avx2   = NULL;    /* resultant polynomial from avx2 mult */

    /* ---------------- Allocate memory buffers ---------------- */
    /* Allign */
    size_t size_a   = ROUND_UP_8(deg_a   + 1    ); /* number of elements of poly_a */
    size_t size_b   = ROUND_UP_8(deg_b   + 1    ); /* number of elements of poly_b */
    size_t size_res = ROUND_UP_8(deg_res + 1 + 8); /* number of elements of poly_res */

    size_t sizeof_a   = size_a   * sizeof(int); /* size in bytes of poly_a */
    size_t sizeof_b   = size_b   * sizeof(int); /* size in bytes of poly_b */
    size_t sizeof_res = size_res * sizeof(int); /* size in bytes of poly_res */

    /* for the input polynomials */
    poly_a = (int*)aligned_alloc(32, sizeof_a); CHECK_MALLOC(poly_a);
    poly_b = (int*)aligned_alloc(32, sizeof_b); CHECK_MALLOC(poly_b);

    /* for the scalar and simd results */
    poly_res_scalar = (int*)aligned_alloc(32, sizeof_res); CHECK_MALLOC(poly_res_scalar);
    poly_res_avx2   = (int*)aligned_alloc(32, sizeof_res); CHECK_MALLOC(poly_res_avx2  );
    /* Initialize output buffers to 0 */
    memset(poly_res_scalar, 0, sizeof_res);
    memset(poly_res_avx2  , 0, sizeof_res);

    /* =========================== Generate the two polynomials =========================== */
    printf("Multiplication of two %ld-degree polynomials.\n", deg_a);
    printf("================================================");
    printf("\nGenerating Polynomials...\n");
    
    srand((unsigned int) time(NULL));   /* seed random generator */
    int max_coeff = MAX_COEFF;          /* maximum coefficient value (absolute value) */

    /* Random fill input polynomials */
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
        poly_random_fill(poly_a, deg_a, max_coeff);
        poly_random_fill(poly_b, deg_b, max_coeff);
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    gen_time = time_diff(&start, &finish); /* elapsed time */
    printf("  Polynomials random fill time    (s): %9.6f\n", gen_time);
    

    /* =========================== Warm up Runs =========================== */
    printf("================================================");
    printf("\nWarm up runs...\n");
    poly_mult_scalar(poly_a, deg_a, poly_b, deg_b, poly_res_scalar, &scalar_time);
    printf("  Scalar poly mult execution time (s): %9.6f\n", scalar_time);
    poly_mult_avx2(poly_a, deg_a, poly_b, deg_b, poly_res_avx2, &simd_time);
    printf("  AVX2 poly mult execution time   (s): %9.6f\n", simd_time);
    
    /* =========================== Scalar Poly Multiplication =========================== */
    printf("================================================");
    printf("\nScalar Poly Multiplication...\n");

    /* Compute Scalar */
    poly_mult_scalar(poly_a, deg_a, poly_b, deg_b, poly_res_scalar, &scalar_time);

    /* Print execution time */
    printf("  Scalar poly mult execution time (s): %9.6f\n", scalar_time);

    #ifdef DEBUG
    print_poly(poly_a, size_a);
    print_poly(poly_b, size_b);
    print_poly(poly_res_scalar, size_res);
    #endif


    /* ============================ AVX2 Poly Multiplication ============================ */
    printf("================================================");
    printf("\nSIMD Poly Multiplication\n");
    
    /* Compute SIMD */
    poly_mult_avx2(poly_a, deg_a, poly_b, deg_b, poly_res_avx2, &simd_time);

    /* Print execution time */
    printf("  AVX2 poly mult execution time   (s): %9.6f\n", simd_time);

    /* ------------------ Speedup calculation ------------------ */
    printf("                            Speedup:   %9.3f", scalar_time/simd_time);
    printf("\n");
    
    /* ------------------------- Confirm correctness ------------------------- */
    printf("================================================");
    printf("\nComparing Scalar & AVX2 poly mult results...\n");
    size_t nerrors;
    nerrors = poly_count_errors(poly_res_avx2, poly_res_scalar, deg_res);
    if (nerrors == 0) {
        printf("  Results match!\n");
    } else {
        printf("  ERROR: Results mismatch! # of errors = %ld\n", nerrors);
    }

    #ifdef DEBUG
    print_poly(poly_a, size_a);
    print_poly(poly_b, size_b);
    print_poly(poly_res_scalar, size_res);
    print_poly(poly_res_avx2, size_res);
    #endif
    

    /* ==================================== Cleanup ==================================== */
    /* Free allocated memory */
    free(poly_res_avx2);
    free(poly_res_scalar);
    free(poly_a);
    free(poly_b);

    return 0;
} /* main */

/*--------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Print a message indicating how program should be started
 *            and terminate.
 */
void Usage(char *prog_name, const long long * const degree_input) {
    fprintf(stderr, "Usage: %s <degree>\n", prog_name);
    fprintf(stderr, "   degree: Degree of the polynomials. Must be positive.\n");
    if (degree_input)
        fprintf(stderr, "           Degree given: %lld\n", *degree_input);

    /* exit */
    exit(0);
} /* Usage */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> /* Required for strerror */
#include <errno.h>  /* Required for errno */

#include <immintrin.h> /* The header for all SIMD intrinsics */
/* Helper macro to round up to multiple of 8 */
#define ROUND_UP_8(n) (((n) + 7) & ~7)

#include "poly_util.h"
#include "poly_random_fill.h"
#include "poly_mult_serial.h"
#include "poly_mult_avx2.h"

/* Helper macro for checking pointer after malloc */
#define CHECK_MALLOC(ptr) \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Fatal Error at %s:%d: %s\n", __FILE__, __LINE__, strerror(errno)); \
        exit(EXIT_FAILURE); \
    }

/* declare helper functions */
double time_diff(struct timespec *start, struct timespec *finish);
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
    double serial_time = 0.;
    double simd_time   = 0.;

    /* polynomial pointers */
    int *poly_a          = NULL;    /* input polynomial */
    int *poly_b          = NULL;    /* input polynomial */
    int *poly_res_serial = NULL;    /* resultant polynomial from serial */
    int *poly_res_simd   = NULL;    /* resultant polynomial from SIMD */

    /* ---------------- Allocate memory buffers ---------------- */
    /* for the input polynomials */
    poly_a = malloc((deg_a + 1) * sizeof(int)); CHECK_MALLOC(poly_a);
    poly_b = malloc((deg_b + 1) * sizeof(int)); CHECK_MALLOC(poly_b);

    /* for the serial and simd results */
    poly_res_serial = calloc(deg_res + 1, sizeof(int)); CHECK_MALLOC(poly_res_serial);
    poly_res_simd   = calloc(deg_res + 1, sizeof(int)); CHECK_MALLOC(poly_res_simd  );
    

    /* =========================== Generate the two polynomials =========================== */
    printf("Multiplication of two %ld-degree polynomials.\n", deg_a);
    printf("================================================");
    printf("\nGenerating Polynomials...\n");
    
    srand((unsigned int) time(NULL));   /* seed random generator */
    int max_coeff = 10;                 /* maximum coefficient value (absolute value) */

    /* Random fill input polynomials */
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
        poly_random_fill(poly_a, deg_a, max_coeff);
        poly_random_fill(poly_b, deg_b, max_coeff);
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    gen_time = time_diff(&start, &finish); /* elapsed time */
    printf("  Polynomials random fill time    (s): %9.6f\n", gen_time);
    
    
    /* =========================== Serial Poly Multiplication =========================== */
    printf("================================================");
    printf("\nSerial Poly Multiplication...\n");

    /* Compute Serial */
    poly_mult_serial(poly_a, deg_a, poly_b, deg_b, poly_res_serial, &serial_time);

    /* Print execution time */
    printf("  Serial poly mult execution time (s): %9.6f\n", serial_time);

    #ifdef DEBUG
    print_poly(poly_a, deg_a);
    print_poly(poly_b, deg_b);
    print_poly(poly_res_serial, deg_res);
    #endif


    /* ============================ SIMD Poly Multiplication ============================ */
    printf("================================================");
    printf("\nSIMD Poly Multiplication\n");
    
    /* Compute SIMD */
    poly_mult_avx2(poly_a, deg_a, poly_b, deg_b, poly_res_simd, &simd_time);

    /* Print execution time */
    printf("  SIMD poly mult execution time   (s): %9.6f\n", simd_time);

    /* ------------------ Speedup calculation ------------------ */
    printf("                            Speedup:   %9.3f", serial_time/simd_time);
    printf("\n");
    
    /* ------------------------- Confirm correctness ------------------------- */
    printf("================================================");
    printf("\nComparing Serial & SIMD poly mult results...\n");
    long long nerrors;
    nerrors = poly_count_errors(poly_res_simd, poly_res_serial, deg_res);
    if (nerrors == 0) {
        printf("  Results match!\n");
    } else {
        printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
    }
    

    /* ==================================== Cleanup ==================================== */
    /* Free allocated memory */
    free(poly_res_simd);
    free(poly_res_serial);
    free(poly_a);
    free(poly_b);

    return 0;
} /* main */



/*--------------------------------------------------------------------
 * Function:  time_diff
 * Purpose:   Returns the elapsed time between the start and the finish
 *            in seconds (with nanosecond accuracy).
 */
double time_diff(struct timespec *start, struct timespec *finish) {
    double time_elapsed = (finish->tv_sec - start->tv_sec) + (finish->tv_nsec - start->tv_nsec) / 1e9; 
    return time_elapsed;
} /* time_diff*/



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
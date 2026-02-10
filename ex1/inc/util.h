#ifndef _util_h
#define _util_h

#include <stdio.h>
#include <time.h>

/* Helper macro to round up to multiple of 8 */
#define ROUND_UP_8(n) (((n) + 7) & ~7)

/* Helper macro for checking pointer after malloc */
#define CHECK_MALLOC(ptr) \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Fatal Error at %s:%d: %s\n", __FILE__, __LINE__, strerror(errno)); \
        exit(EXIT_FAILURE); \
    }


double time_delta(struct timespec *start, struct timespec *finish);
double get_min_double(double val_a, double val_b);


#endif
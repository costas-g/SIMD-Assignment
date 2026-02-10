#include <time.h>

/*--------------------------------------------------------------------
 * Function:  time_delta
 * Purpose:   Returns the elapsed time between the start and the finish
 *            in seconds (with nanosecond accuracy).
 */
double time_delta(struct timespec *start, struct timespec *finish) {
    double time_elapsed = (finish->tv_sec - start->tv_sec) + (finish->tv_nsec - start->tv_nsec) / 1e9; 
    return time_elapsed;
} /* time_delta*/


/*--------------------------------------------------------------------
 * Function:  get_min_double
 * Purpose:   Returns the minimum value of the two double arguments
 */
double get_min_double(double val_a, double val_b) {
    if(val_a < val_b)
        return val_a;
    else
        return val_b;
} /* get_min_double*/
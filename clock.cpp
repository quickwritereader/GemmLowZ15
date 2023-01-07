
#include <time.h>

double FLA_Clock_helper(  );

// A global variable used when FLA_Clock_helper() is defined in terms of
// clock_gettime()/gettimeofday().
double gtod_ref_time_sec = 0.0;

double FLA_Clock(  )
{
	return FLA_Clock_helper();
}



double FLA_Clock_helper()
{
    double the_time, norm_sec;
    struct timespec ts;

    clock_gettime( CLOCK_MONOTONIC, &ts );

    if ( gtod_ref_time_sec == 0.0 )
        gtod_ref_time_sec = ( double ) ts.tv_sec;

    norm_sec = ( double ) ts.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + ts.tv_nsec * 1.0e-9;

    return the_time;
}




/*
 * cball_wrapper.c ? ctypes-compatible wrapper for Ball C kernel functions
 * Compile:
 *   Windows: gcc -shared -O3 -fopenmp -o cball_ext.dll cball_wrapper.c bdd_matrix.c utilities.c median.c -lm
 *   Linux:   gcc -shared -O3 -fopenmp -o cball_ext.so  cball_wrapper.c bdd_matrix.c utilities.c median.c -lm -fPIC
 */

#include "bdd_matrix.h"

#if defined(_WIN32) || defined(_WIN64)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/*
 * Compute BDD kernel matrix from a distance vector.
 *
 * Parameters:
 *   b_dd       ? output: packed lower-triangle of BDD matrix [n*(n+1)/2]
 *   dist_vec   ? input:  packed upper-triangle of distance matrix [n*(n-1)/2]
 *   n          ? sample size
 *   nthread    ? number of OpenMP threads (0 = auto, 1 = single-thread)
 *   weight_type ? 1=constant, 2=probability, 3=chisquare, 4=rbf
 */
EXPORT void compute_bdd_bias(double *b_dd, double const *dist_vec,
                              int *n, int *nthread, int *weight_type)
{
    /* bdd_matrix_bias expects non-const input; cast away const safely */
    bdd_matrix_bias(b_dd, (double *)dist_vec, n, nthread, weight_type);
}

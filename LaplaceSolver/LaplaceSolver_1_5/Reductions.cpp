#include "Reductions.h"
#include <algorithm>
#include <iostream>

#define DO_NOT_USE_MKL
#ifndef DO_NOT_USE_MKL
#include <mkl.h>
#endif

float Norm(const float (&x)[XDIM][YDIM][ZDIM])
{
    float result = 0.;
    int index = 0;
    const float* x_flat = reinterpret_cast<const float*>(x);
        
#ifdef DO_NOT_USE_MKL    
#pragma omp parallel for reduction(max:result)
//    for (int i = 1; i < XDIM-1; i++)
//    for (int j = 1; j < YDIM-1; j++)
//    for (int k = 1; k < ZDIM-1; k++)
      for(int i=0; i<(XDIM*YDIM*ZDIM); i++)
	result = std::max(result, std::abs(x_flat[i]));
#endif

#ifndef DO_NOT_USE_MKL
    index = cblas_isamax(XDIM*YDIM*ZDIM,
                         &x_flat[0],
	                 1U);
    result = x_flat[index-1];
#endif

    return result;
}

float InnerProduct(const float (&x)[XDIM][YDIM][ZDIM], const float (&y)[XDIM][YDIM][ZDIM])
{
    double result = 0.;

#ifdef DO_NOT_USE_MKL
#pragma omp parallel for reduction(+:result)
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        result += (double) x[i][j][k] * (double) y[i][j][k];
#endif
#ifndef DO_NOT_USE_MKL
    result = cblas_sdsdot(XDIM*YDIM*ZDIM,
		           0U,
		           &x[0][0][0],
			   1U,
			   &y[0][0][0],
			   1U);
#endif
    
    return (float) result;
}

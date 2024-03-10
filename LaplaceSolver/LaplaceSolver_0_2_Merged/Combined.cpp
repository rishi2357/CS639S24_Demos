#include "Parameters.h"

#include <algorithm>
#include <iostream>

/* Passing p, x, r */
void CombinedSaxpy(float (&p)[XDIM][YDIM][ZDIM], float (&x)[XDIM][YDIM][ZDIM], const float (&r)[XDIM][YDIM][ZDIM],
    const float alpha, const float beta)
{
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
    {
        x[i][j][k] = p[i][j][k] * alpha + x[i][j][k];
        p[i][j][k] = p[i][j][k] * beta  + r[i][j][k];
    }
}

/* Passing r, z */
float CombinedCopyIP(const float (&r)[XDIM][YDIM][ZDIM], float (&z)[XDIM][YDIM][ZDIM])
{
    double result = 0.;

#pragma omp parallel for    
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        z[i][j][k] = r[i][j][k];

#pragma omp parallel for reduction(+:result)
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        result += (double) r[i][j][k] * (double) z[i][j][k];

    return (float) result;
}

/* Passing p(u), z(Lu) */
float Combined_CL_IP(float (&u)[XDIM][YDIM][ZDIM], float (&Lu)[XDIM][YDIM][ZDIM])
{
    double result = 0.;

#pragma omp parallel for
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        Lu[i][j][k] =
            -6 * u[i][j][k]
            + u[i+1][j][k]
            + u[i-1][j][k]
            + u[i][j+1][k]
            + u[i][j-1][k]
            + u[i][j][k+1]
            + u[i][j][k-1];

#pragma omp parallel for reduction(+:result)
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        result += (double) u[i][j][k] * (double) Lu[i][j][k];
    
    return (float) result;
}
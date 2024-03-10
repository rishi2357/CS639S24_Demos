#pragma once

#include "Parameters.h"

void CombinedSaxpy(float (&p)[XDIM][YDIM][ZDIM], float (&x)[XDIM][YDIM][ZDIM], const float (&r)[XDIM][YDIM][ZDIM],
    const float alpha, const float beta);

float CombinedCopyIP(const float (&r)[XDIM][YDIM][ZDIM], float (&z)[XDIM][YDIM][ZDIM]);

float Combined_CL_IP(float (&u)[XDIM][YDIM][ZDIM], float (&Lu)[XDIM][YDIM][ZDIM]);
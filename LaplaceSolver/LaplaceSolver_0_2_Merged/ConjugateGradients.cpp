#include "Laplacian.h"
#include "Parameters.h"
#include "PointwiseOps.h"
#include "Reductions.h"
#include "Utilities.h"
#include "Timer.h"
#include "Combined.h"

#include <iostream>

using elapsed_time_t = std::chrono::duration<double, std::milli>;
extern Timer timerKernel;

/*  Time capture for non-iterative function
    calls. */
extern elapsed_time_t time_ComputeLaplacianSingle;
extern elapsed_time_t time_SaxpySingle;
extern elapsed_time_t time_NormSingle;
extern elapsed_time_t time_CopySingle;
extern elapsed_time_t time_InnerProductSingle;

/*  Time capture for iterative calls. */
extern elapsed_time_t time_ComputerLaplacianLoop[260];
extern elapsed_time_t time_SaxpyLoop[4][260];
extern elapsed_time_t time_NormLoop[260];
extern elapsed_time_t time_CopyLoop[260];
extern elapsed_time_t time_InnerProductLoop[2][260];
extern elapsed_time_t time_CombinedLoop[260];

void ConjugateGradients(
    float (&x)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM],
    const bool writeIterations)
{
    // Algorithm : Line 2
    timerKernel.Restart(); ComputeLaplacian(x, z); timerKernel.Pause();
    time_ComputeLaplacianSingle += timerKernel.mElapsedTime;

    timerKernel.Restart(); Saxpy(z, f, r, -1); timerKernel.Pause();
    time_SaxpySingle += timerKernel.mElapsedTime;
    
    timerKernel.Restart(); float nu = Norm(r); timerKernel.Pause();
    time_NormSingle += timerKernel.mElapsedTime;

    // Algorithm : Line 3
    if (nu < nuMax) return;
        
    // Algorithm : Line 4
    timerKernel.Restart(); Copy(r, p); timerKernel.Pause();
    time_CopySingle += timerKernel.mElapsedTime;
    
    timerKernel.Restart(); float rho=InnerProduct(p, r); timerKernel.Pause();
    time_InnerProductSingle += timerKernel.mElapsedTime;

    // Beginning of loop from Line 5
    for(int k=0;;k++)
    {
        //std::cout << "Residual norm (nu) after " << k << " iterations = " << nu << std::endl;

        // Algorithm : Line 6
        // timerKernel.Restart(); ComputeLaplacian(p, z); timerKernel.Pause();
        // time_ComputerLaplacianLoop[k] = timerKernel.mElapsedTime;

        // timerKernel.Restart(); float sigma=InnerProduct(p, z); timerKernel.Pause();
        // time_InnerProductLoop[0][k] = timerKernel.mElapsedTime;

        timerKernel.Restart(); float sigma = Combined(p, z, r, r, rho); timerKernel.Pause();
        time_CombinedLoop[k] = timerKernel.mElapsedTime;

        // // Algorithm : Line 7
        float alpha=rho/sigma;

        // // Algorithm : Line 8
        timerKernel.Restart(); Saxpy(z, r, r, -alpha); timerKernel.Pause();
        time_SaxpyLoop[0][k] = timerKernel.mElapsedTime;
        
        timerKernel.Restart(); nu=Norm(r); timerKernel.Pause();
        time_NormLoop[k] = timerKernel.mElapsedTime;

        // timerKernel.Restart(); float nu, alpha = Combined(p, z, r, r, rho); timerKernel.Pause();
        // time_CombinedLoop[k] = timerKernel.mElapsedTime;

        // Algorithm : Lines 9-12
        if (nu < nuMax || k == kMax) {
            timerKernel.Restart(); Saxpy(p, x, x, alpha); timerKernel.Pause();
            time_SaxpyLoop[3][k] = timerKernel.mElapsedTime;
            std::cout << "Conjugate Gradients terminated after " << k << " iterations; residual norm (nu) = " << nu << std::endl;
            if (writeIterations) WriteAsImage("x", x, k, 0, 127);
            return;
        }
            
        // Algorithm : Line 13
        timerKernel.Restart(); Copy(r, z); timerKernel.Pause();
        time_CopyLoop[k] = timerKernel.mElapsedTime;

        timerKernel.Restart(); float rho_new = InnerProduct(z, r); timerKernel.Pause();
        time_InnerProductLoop[0][k] = timerKernel.mElapsedTime;

        // Algorithm : Line 14
        float beta = rho_new/rho;

        // Algorithm : Line 15
        rho=rho_new;

        // Algorithm : Line 16
        timerKernel.Restart(); Saxpy(p, x, x, alpha); timerKernel.Pause();
        time_SaxpyLoop[1][k] = timerKernel.mElapsedTime;

        timerKernel.Restart(); Saxpy(p, r, p, beta); timerKernel.Pause();
        time_SaxpyLoop[2][k] = timerKernel.mElapsedTime;

        if (writeIterations) WriteAsImage("x", x, k, 0, 127);
    }
}

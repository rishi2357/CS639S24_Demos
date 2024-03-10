#include "ConjugateGradients.h"
#include "Timer.h"
#include "Utilities.h"
#include "Reductions.h"

#include <chrono>

Timer timerKernel;

using elapsed_time_t = std::chrono::duration<double, std::milli>;

/*  Time capture for non-iterative function
    calls. */
elapsed_time_t time_ComputeLaplacianSingle;
elapsed_time_t time_SaxpySingle;
elapsed_time_t time_NormSingle;
elapsed_time_t time_CopySingle;
elapsed_time_t time_InnerProductSingle;

/*  Time capture for iterative calls. */
elapsed_time_t time_ComputerLaplacianLoop[260];
elapsed_time_t time_SaxpyLoop[4][260];
elapsed_time_t time_NormLoop[260];
elapsed_time_t time_CopyLoop[260];
elapsed_time_t time_InnerProductLoop[2][260];
elapsed_time_t time_CombinedLoop[260];

/* Cumulative time capture */
double time_ComputeLaplacian;
double time_Saxpy;
double time_Norm;
double time_Copy;
double time_InnerProduct;
double time_ComputedTotal;
double time_Combined;

int main(int argc, char *argv[])
{
    using array_t = float (&) [XDIM][YDIM][ZDIM];

    float *xRaw = new float [XDIM*YDIM*ZDIM];
    float *fRaw = new float [XDIM*YDIM*ZDIM];
    float *pRaw = new float [XDIM*YDIM*ZDIM];
    float *rRaw = new float [XDIM*YDIM*ZDIM];
    float *zRaw = new float [XDIM*YDIM*ZDIM];
    
    array_t x = reinterpret_cast<array_t>(*xRaw);
    array_t f = reinterpret_cast<array_t>(*fRaw);
    array_t p = reinterpret_cast<array_t>(*pRaw);
    array_t r = reinterpret_cast<array_t>(*rRaw);
    array_t z = reinterpret_cast<array_t>(*zRaw);
    
    /* Initialize time capture variables. */
    time_ComputedTotal                = double(0.);
    time_ComputeLaplacianSingle       = elapsed_time_t(0.);
    time_SaxpySingle                  = elapsed_time_t(0.);
    time_NormSingle                   = elapsed_time_t(0.);
    time_CopySingle                   = elapsed_time_t(0.);
    time_InnerProductSingle           = elapsed_time_t(0.);
    time_ComputerLaplacianLoop[260]   = elapsed_time_t({});
    time_SaxpyLoop[4][260]            = elapsed_time_t({});
    time_NormLoop[260]                = elapsed_time_t({});
    time_CopyLoop[260]                = elapsed_time_t({});
    time_InnerProductLoop[2][260]     = elapsed_time_t({});
    
    // Initialization
    {
        Timer timer;
        timer.Start();
        InitializeProblem(x, f);
        timer.Stop("Initialization : ");
    }

    // Call Conjugate Gradients algorithm
    timerKernel.Reset();
    ConjugateGradients(x, f, p, r, z);

    /* Print non-iterative function call durations */
    std::cout << "Non-iterative ComputeLaplacian time = " << time_ComputeLaplacianSingle.count() << "ms" << std::endl;
    std::cout << "Non-iterative Saxpy time = " << time_SaxpySingle.count() << "ms" << std::endl;
    std::cout << "Non-iterative Norm time = " << time_NormSingle.count() << "ms" << std::endl;
    std::cout << "Non-iterative Copy time = " << time_CopySingle.count() << "ms" << std::endl;
    std::cout << "Non-iterative Inner Product time = " << time_InnerProductSingle.count() << "ms" << std::endl;

    time_ComputedTotal = time_ComputeLaplacianSingle.count() + time_SaxpySingle.count() + time_NormSingle.count() +
                         time_CopySingle.count() + time_InnerProductSingle.count();

    /* Compute cumulative time for iterated function call durations */
    /* Combined Kernel */
    time_Combined = double(0.);
    #pragma omp parallel for reduction(+:time_ComputeLaplacian)
    for(int i=0; i<260; i++)
        time_Combined += time_CombinedLoop[i].count();

    std::cout << "Cumulative Combined Kernel time across all iterations = " << time_Combined << "ms" << std::endl;

    time_ComputedTotal += time_Combined;

    /* 1st SAXPY call */
    time_Saxpy = double(0.);
    #pragma omp parallel for reduction(+:time_Saxpy)
    for(int i=0; i<260; i++)
        time_Saxpy += time_SaxpyLoop[0][i].count();

    std::cout << "Cumulative 1st Saxpy time across all iterations = " << time_Saxpy << "ms" << std::endl;

    time_ComputedTotal += time_Saxpy;

    /* 2nd SAXPY call */
    time_Saxpy = double(0.);
    #pragma omp parallel for reduction(+:time_Saxpy)
    for(int i=0; i<260; i++)
        time_Saxpy += time_SaxpyLoop[1][i].count();

    std::cout << "Cumulative 2nd Saxpy time across all iterations = " << time_Saxpy << "ms" << std::endl;

    time_ComputedTotal += time_Saxpy;

    /* 3rd SAXPY call */
    time_Saxpy = double(0.);
    #pragma omp parallel for reduction(+:time_Saxpy)
    for(int i=0; i<260; i++)
        time_Saxpy += time_SaxpyLoop[2][i].count();

    std::cout << "Cumulative 3rd Saxpy time across all iterations = " << time_Saxpy << "ms" << std::endl;

    time_ComputedTotal += time_Saxpy;

    /* 4th SAXPY call */
    time_Saxpy = double(0.);
    #pragma omp parallel for reduction(+:time_Saxpy)
    for(int i=0; i<260; i++)
        time_Saxpy += time_SaxpyLoop[3][i].count();

    std::cout << "Cumulative 4th Saxpy time across all iterations = " << time_Saxpy << "ms" << std::endl;

    time_ComputedTotal += time_Saxpy;

    /* Norm */
    time_Norm = double(0.);
    #pragma omp parallel for reduction(+:time_Norm)
    for(int i=0; i<260; i++)
        time_Norm += time_NormLoop[i].count();

    std::cout << "Cumulative Norm time across all iterations = " << time_Norm << "ms" << std::endl;

    time_ComputedTotal += time_Norm;

    /* Copy */
    time_Copy = double(0.);
    #pragma omp parallel for reduction(+:time_Copy)
    for(int i=0; i<260; i++)
        time_Copy += time_CopyLoop[i].count();

    std::cout << "Cumulative Copy time across all iterations = " << time_Copy << "ms" << std::endl;

    time_ComputedTotal += time_Copy;

    /* 1st InnerProduct call */
    time_InnerProduct = double(0.);
    #pragma omp parallel for reduction(+:time_InnerProduct)
    for(int i=0; i<260; i++)
        time_InnerProduct += time_InnerProductLoop[0][i].count();

    std::cout << "Cumulative 1st Inner Product time across all iterations = " << time_InnerProduct << "ms" << std::endl;

    time_ComputedTotal += time_InnerProduct;

    /* 2nd InnerProduct call */
    time_InnerProduct = double(0.);
    #pragma omp parallel for reduction(+:time_InnerProduct)
    for(int i=0; i<260; i++)
        time_InnerProduct += time_InnerProductLoop[1][i].count();

    std::cout << "Cumulative 2nd Inner Product time across all iterations = " << time_InnerProduct << "ms" << std::endl;

    time_ComputedTotal += time_InnerProduct;

    std::cout << "Cumulative computed time = " << time_ComputedTotal << "ms" << std::endl;

    timerKernel.Print("Total Conjugate Gradient Measured Time: ");

    return 0;
}

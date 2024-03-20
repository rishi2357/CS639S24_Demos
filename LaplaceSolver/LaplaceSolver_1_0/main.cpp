#include "ConjugateGradients.h"
#include "Laplacian.h"
#include "Timer.h"
#include "Utilities.h"

Timer timerIteration;
Timer timerLaplacian;
Timer timerSaxpy;
Timer timerCopy;
Timer timerInnerProduct;
Timer timerConjugateGradient;

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
    
    CSRMatrix matrix;

    // Initialization
    {
        Timer timer;
        timer.Start();
        InitializeProblem(x, f);
        matrix = BuildLaplacianMatrix(); // This takes a while ...
        timer.Stop("Initialization : ");
    }

    // Call Conjugate Gradients algorithm
    timerIteration.Reset(); timerConjugateGradient.Reset();
    timerLaplacian.Reset(); timerSaxpy.Reset();
    timerCopy.Reset(); timerInnerProduct.Reset();

    timerConjugateGradient.Restart();
    ConjugateGradients(matrix, x, f, p, r, z, false);
    timerConjugateGradient.Pause();

    timerConjugateGradient.Print("Total CG Time : ");
    timerLaplacian.Print("Total Laplacian Time : ");
    timerSaxpy.Print("Total Saxpy Time : ");
    timerCopy.Print("Total Copy Time : ");
    timerInnerProduct.Print("Total Inner Product Time : ");

    return 0;
}

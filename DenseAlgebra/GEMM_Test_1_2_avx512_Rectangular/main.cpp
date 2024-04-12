#include "MatMatMultiply.h"
#include "Timer.h"
#include "Utilities.h"

#include <iostream>
#include <iomanip>

int main(int argc, char *argv[])
{
    /* Initialize rectangular matrices */
    float *Araw = static_cast<float*>( AlignedAllocate( MATRIX_SIZE_P * MATRIX_SIZE_L * sizeof(float), 64 ) );
    float *Braw = static_cast<float*>( AlignedAllocate( MATRIX_SIZE_L * MATRIX_SIZE_Q * sizeof(float), 64 ) );
    float *Craw = static_cast<float*>( AlignedAllocate( MATRIX_SIZE_P * MATRIX_SIZE_Q * sizeof(float), 64 ) );
    float *referenceCraw = static_cast<float*>( AlignedAllocate( MATRIX_SIZE_P * MATRIX_SIZE_Q * sizeof(float), 64 ) );

    using matrix_t_A = float (&) [MATRIX_SIZE_P][MATRIX_SIZE_L];
    using matrix_t_B = float (&) [MATRIX_SIZE_L][MATRIX_SIZE_Q];
    using matrix_t_C = float (&) [MATRIX_SIZE_P][MATRIX_SIZE_Q];
    
    matrix_t_A A = reinterpret_cast<matrix_t_A>(*Araw);
    matrix_t_B B = reinterpret_cast<matrix_t_B>(*Braw);
    matrix_t_C C = reinterpret_cast<matrix_t_C>(*Craw);
    matrix_t_C referenceC = reinterpret_cast<matrix_t_C>(*referenceCraw);

    InitializeMatrices(A, B);
    Timer timer;

    // Correctness test
    std::cout << "Running candidate kernel for correctness test ... " << std::flush;
    timer.Start();
    MatMatMultiply(A, B, C);
    timer.Stop("Elapsed time : ");
    
    std::cout << "Running reference kernel for correctness test ... " << std::flush;
    timer.Start();
    MatMatMultiplyReference(A, B, referenceC);
    timer.Stop("Elapsed time : ");

    float discrepancy = MatrixMaxDifference(C, referenceC);
    std::cout << "Discrepancy between two methods : " << discrepancy << std::endl;
    
    for(int test = 1; test <= 20; test++)
    {
        std::cout << "Running kernel for performance run #" << std::setw(2) << test << " ... ";
        timer.Start();
        MatMatMultiply(A, B, C);
        timer.Stop("Elapsed time : ");
    }
    
    return 0;
}

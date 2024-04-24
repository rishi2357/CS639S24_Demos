#include "DirectSolver.h"
#include "Utilities.h"

#include "mkl.h"

#include <iostream>
#include <memory>
#include <numeric>
#include <array>

void DirectSparseSolver(
    CSRMatrix& matrix,
    float (&x)[XDIM][YDIM][ZDIM],
    float (&f)[XDIM][YDIM][ZDIM],
    const bool writeOutput)
{
    const MKL_INT n = matrix.mSize; // Matrix size
    MKL_INT mtype = 2;        // Real symmetric positive definite matrix
    const MKL_INT nrhs = 100;  // Number of right hand sides
    void *pt[64];             // Internal solver memory pointer pt
                              // should be "int" when using 32-bit architectures, or "long int"
                              // for 64-bit architectures. void* should be OK in both cases
    MKL_INT iparm[64];        // Pardiso control parameters
    MKL_INT maxfct, mnum, phase, error, msglvl;
    MKL_INT rhs, i,j,k;
    float ddum;               // Scalar dummy (PARDISO needs it)
    MKL_INT idum;             // Integer dummy (PARDISO needs it)

    //std::vector<std::vector<std::vector<float>>> bMultiple[nrhs];
    //std::vector<std::vector<std::vector<float>>> xMultiple[nrhs];
    using array_4d = std::array<std::array<std::array<std::array<float, ZDIM>, YDIM>, XDIM>,\
                     std::size_t(nrhs)>;

    std::unique_ptr<array_4d> bMultiplePtr = std::make_unique<array_4d>();
    std::unique_ptr<array_4d> xMultiplePtr = std::make_unique<array_4d>();

    array_4d& bMultiple = *bMultiplePtr;
    array_4d& xMultiple = *xMultiplePtr;

    // Fill the vector with consecutive values starting from 0
    //std::iota(perm.begin(), perm.end(), 0);
    /* Make multiple right-hand sides */
    for (rhs=0; rhs<nrhs; rhs++)
        for(i=0; i<XDIM; i++)
            for(j=0; j<YDIM; j++)
                for(k=0; k<ZDIM; k++)
                    bMultiple[rhs][i][j][k] = f[i][j][k];

    /* Make multiple unknowns(x) */
    for (rhs=0; rhs<nrhs; rhs++)
        for(i=0; i<XDIM; i++)
            for(j=0; j<YDIM; j++)
                for(k=0; k<ZDIM; k++)
                    xMultiple[rhs][i][j][k] = x[i][j][k];

    // Set-up PARDISO control parameters
    for ( i = 0; i < 64; i++ )
    {
        iparm[i] = 0;
    }
    iparm[0] = 1;         // No solver default
    iparm[1] = 3;         // Parallel nested dissection
    iparm[3] = 0;         // No iterative-direct algorithm
    iparm[4] = 0;         // No user fill-in reducing permutation
    iparm[5] = 0;         // Write solution into x
    iparm[6] = 0;         // Not in use
    iparm[7] = 0;         // Max numbers of iterative refinement steps
    iparm[8] = 0;         // Not in use
    iparm[9] = 13;        // Perturb the pivot elements with 1E-13
    iparm[10] = 1;        // Use nonsymmetric permutation and scaling MPS
    iparm[11] = 0;        // Not in use
    iparm[12] = 0;        // Maximum weighted matching algorithm is switched-off
                          // Try iparm[12] = 1 in case of inappropriate accuracy
    iparm[13] = 0;        // Output: Number of perturbed pivots
    iparm[14] = 0;        // Not in use
    iparm[15] = 0;        // Not in use
    iparm[16] = 0;        // Not in use
    iparm[17] = -1;       // Output: Number of nonzeros in the factor LU
    iparm[18] = -1;       // Output: Mflops for LU factorization
    iparm[19] = 0;        // Output: Numbers of CG Iterations
    iparm[23] = 1;        // Two-level factorization*/
    iparm[26] = 1;        // Check matrix for errors
    iparm[27] = 1;        // Use float precision
    iparm[34] = 1;        // Use zero-based indexing
    maxfct = 1;           // Maximum number of numerical factorizations.
    mnum = 1;             // Which factorization to use.
    msglvl = 1;           // Print statistical information in file
    error = 0;            // Initialize error flag

    // Initialize the internal solver memory pointer. This is only
    // necessary for the FIRST call of the PARDISO solver
    for ( i = 0; i < 64; i++ )
    {
        pt[i] = 0;
    }

    // Reordering and Symbolic Factorization. This step also allocates
    // all memory that is necessary for the factorization
    phase = 11;
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase, &n,
        matrix.GetValues(), matrix.GetRowOffsets(), matrix.GetColumnIndices(),
        &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 )
        throw std::runtime_error("PARDISO error during symbolic factorization");

    std::cout << "Reordering completed ... " << std::endl;
    std::cout << "Number of nonzeros in factors = " <<  iparm[17] << std::endl;
    std::cout << "Number of factorization MFLOPS = " << iparm[18] << std::endl;

    // Numerical factorization
    phase = 22;
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase, &n,
        matrix.GetValues(), matrix.GetRowOffsets(), matrix.GetColumnIndices(),
        &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 )
        throw std::runtime_error("PARDISO error during numerical factorization");

    std::cout << "Factorization completed ... " << std::endl;

    // Back substitution and iterative refinement
    phase = 33;
    iparm[7] = 0;         // Max numbers of iterative refinement steps
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase, &n,
        matrix.GetValues(), matrix.GetRowOffsets(), matrix.GetColumnIndices(),
        &idum, &nrhs, iparm, &msglvl, static_cast<void*>(&bMultiple[0][0][0][0]), &xMultiple[0][0][0][0], &error);
    if ( error != 0 )
        throw std::runtime_error("PARDISO error during solution phase");

    std::cout << "Solve completed ... " <<std::endl;

    // Termination and release of memory.
    phase = -1;           // Release internal memory
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase, &n,
        &ddum, matrix.GetRowOffsets(), matrix.GetColumnIndices(),
        &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

    for(rhs=0; rhs<nrhs; rhs++)
        if (writeOutput) WriteAsImage("x", (float (&) [XDIM][YDIM][ZDIM])xMultiple[rhs], 0, 0, XDIM/2);
}

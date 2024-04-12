#include "MatMatMultiply.h"
#include "MatMatMultiplyBlockHelper.h"
#include "mkl.h"

alignas(64) float localA[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localB[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localC[BLOCK_SIZE][BLOCK_SIZE];

#pragma omp threadprivate(localA, localB, localC)

void MatMatMultiply(const float (&A)[MATRIX_SIZE_P][MATRIX_SIZE_L],
    const float (&B)[MATRIX_SIZE_L][MATRIX_SIZE_Q], float (&C)[MATRIX_SIZE_P][MATRIX_SIZE_Q])
{
    /* Unequal blocks due to rectangular matrices */
    static constexpr int NBLOCKS_P = MATRIX_SIZE_P / BLOCK_SIZE;
    static constexpr int NBLOCKS_L = MATRIX_SIZE_L / BLOCK_SIZE;
    static constexpr int NBLOCKS_Q = MATRIX_SIZE_Q / BLOCK_SIZE;

    using blocked_matrix_t = float (&) [NBLOCKS_P][BLOCK_SIZE][NBLOCKS_Q][BLOCK_SIZE];
    using const_blocked_matrix_t_A = const float (&) [NBLOCKS_P][BLOCK_SIZE][NBLOCKS_L][BLOCK_SIZE];
    using const_blocked_matrix_t_B = const float (&) [NBLOCKS_L][BLOCK_SIZE][NBLOCKS_Q][BLOCK_SIZE];

    auto blockA = reinterpret_cast<const_blocked_matrix_t_A>(A[0][0]);
    auto blockB = reinterpret_cast<const_blocked_matrix_t_B>(B[0][0]);
    auto blockC = reinterpret_cast<blocked_matrix_t>(C[0][0]);

#pragma omp parallel for
    for (int bi = 0; bi < NBLOCKS_P; bi++)
    for (int bj = 0; bj < NBLOCKS_Q; bj++) {
        
        for (int ii = 0; ii < BLOCK_SIZE; ii++)
            for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                localC[ii][jj] = 0.;
            }

        for (int bk = 0; bk < NBLOCKS_L; bk++) {

            for (int ii = 0; ii < BLOCK_SIZE; ii++)
            for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                localA[ii][jj] = blockA[bi][ii][bk][jj];
                localB[ii][jj] = blockB[bk][ii][bj][jj];
            }

            MatMatMultiplyBlockHelper(localA, localB, localC);
        }

        for (int ii = 0; ii < BLOCK_SIZE; ii++)
        for (int jj = 0; jj < BLOCK_SIZE; jj++)
            blockC[bi][ii][bj][jj] = localC[ii][jj];
    }
}

void MatMatMultiplyReference(const float (&A)[MATRIX_SIZE_P][MATRIX_SIZE_L],
    const float (&B)[MATRIX_SIZE_L][MATRIX_SIZE_Q], float (&C)[MATRIX_SIZE_P][MATRIX_SIZE_Q])
{
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        MATRIX_SIZE_P,
        MATRIX_SIZE_Q,
        MATRIX_SIZE_L,
        1.,
        &A[0][0],
        MATRIX_SIZE_L,
        &B[0][0],
        MATRIX_SIZE_Q,
        0.,
        &C[0][0],
        MATRIX_SIZE_Q
    );
}

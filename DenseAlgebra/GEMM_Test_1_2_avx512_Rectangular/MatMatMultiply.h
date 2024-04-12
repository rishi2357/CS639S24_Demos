#pragma once

#include "Parameters.h"

void MatMatMultiply(const float (&A)[MATRIX_SIZE_P][MATRIX_SIZE_L],
    const float (&B)[MATRIX_SIZE_L][MATRIX_SIZE_Q], float (&C)[MATRIX_SIZE_P][MATRIX_SIZE_Q]);

void MatMatMultiplyReference(const float (&A)[MATRIX_SIZE_P][MATRIX_SIZE_L],
    const float (&B)[MATRIX_SIZE_L][MATRIX_SIZE_Q], float (&C)[MATRIX_SIZE_P][MATRIX_SIZE_Q]);

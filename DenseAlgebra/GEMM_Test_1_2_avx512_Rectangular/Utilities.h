#pragma once

#include "Parameters.h"

#include <cstdlib>

void* AlignedAllocate(const std::size_t size, const std::size_t alignment);
void InitializeMatrices(float (&A)[MATRIX_SIZE_P][MATRIX_SIZE_L],float (&B)[MATRIX_SIZE_L][MATRIX_SIZE_Q]);
float MatrixMaxDifference(const float (&A)[MATRIX_SIZE_P][MATRIX_SIZE_Q],const float (&B)[MATRIX_SIZE_P][MATRIX_SIZE_Q]);

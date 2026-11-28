#pragma once

#include "matrix_comp.h"

// Target the nroots largest eigenvalues, ordered by their signed values.
// At restart, return true if every requested full-space residual norm is below tol.
// max_iter counts restart cycles; each cycle permits expand_iter expansions.
bool block_search(const Matrix<double>& A, const Matrix<double>& guess, int nroots,
                  double tol, int max_iter,
                  double (*select_shift)(double), int expand_iter, int expand_size);

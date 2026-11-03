#ifndef DAVIDSON_H
#define DAVIDSON_H

#include "matrix.h"

Matrix initialize_vector(int size, const Matrix& M);
// Diagonalize a real symmetric A in place using at most max_iter rotations.
Matrix MatrixJacobi(Matrix& A, int max_iter);

#endif

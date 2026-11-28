#ifndef STRUCTURE_H
#define STRUCTURE_H

#include "matrix_comp.h"
#include <string>
#include <vector>

class Structure{
public:
    Structure(const Matrix<double>& latVec, const Matrix<double>& crysCoord, const std::vector<int>& atomicNum, const Matrix<double>& atomMass);
    ~Structure();

    // Recompute and store reciprocal vectors as rows: a_i . b_j = 2*pi*delta_ij.
    void getRecVec();

    // A 3x3 matrix with one direct lattice vector per row.
    Matrix<double> latVec;
    Matrix<double> crysCoord;
    std::vector<int> atomicNum;
    Matrix<double> recVec;
    Matrix<double> atomMass;
    std::vector<std::string> pseudo;
};

#endif

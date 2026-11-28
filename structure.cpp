#include "structure.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

Structure::Structure(const Matrix<double>& latVec,
                     const Matrix<double>& crysCoord,
                     const std::vector<int>& atomicNum,
                     const Matrix<double>& atomMass)
    : latVec(latVec),
      crysCoord(crysCoord),
      atomicNum(atomicNum),
      recVec(3, 3),
      atomMass(atomMass)
{
    getRecVec();
}

Structure::~Structure() = default;

void Structure::getRecVec() {
    if (latVec.rows() != 3 || latVec.cols() != 3)
        throw std::invalid_argument("Lattice vectors must form a 3x3 matrix");

    // Normalize each row before taking cross products to avoid scale-dependent
    // overflow/underflow and to check degeneracy independently of length units.
    Matrix<double> lengths(3, 1);
    Matrix<double> unit(3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (!std::isfinite(latVec[i][j]))
                throw std::invalid_argument("Lattice vectors must be finite");
        }
        lengths[i][0] = std::hypot(latVec[i][0], latVec[i][1], latVec[i][2]);
        if (lengths[i][0] == 0.0 || !std::isfinite(lengths[i][0]))
            throw std::invalid_argument("Lattice vector lengths must be finite and positive");
        for (int j = 0; j < 3; ++j)
            unit[i][j] = latVec[i][j] / lengths[i][0];
    }

    // Cyclic cross products: u2 x u3, u3 x u1, u1 x u2.
    Matrix<double> cross(3, 3);
    for (int i = 0; i < 3; ++i) {
        const int j = (i + 1) % 3;
        const int k = (i + 2) % 3;
        cross[i][0] = unit[j][1] * unit[k][2] - unit[j][2] * unit[k][1];
        cross[i][1] = unit[j][2] * unit[k][0] - unit[j][0] * unit[k][2];
        cross[i][2] = unit[j][0] * unit[k][1] - unit[j][1] * unit[k][0];
    }

    // Keep the sign: left-handed and right-handed cells must both obey a_i.b_j.
    const double determinant = unit[0][0] * cross[0][0]
                             + unit[0][1] * cross[0][1]
                             + unit[0][2] * cross[0][2];
    if (std::abs(determinant) <= 64.0 * std::numeric_limits<double>::epsilon())
        throw std::invalid_argument("Lattice vectors are singular or nearly linearly dependent");

    const double two_pi = 2.0 * std::acos(-1.0);
    Matrix<double> result(3, 3);
    for (int i = 0; i < 3; ++i) {
        const double factor = (two_pi / lengths[i][0]) / determinant;
        for (int j = 0; j < 3; ++j) {
            result[i][j] = factor * cross[i][j];
            if (!std::isfinite(result[i][j]))
                throw std::overflow_error("Reciprocal lattice vectors exceed floating-point range");
        }
    }

    recVec = std::move(result);
}

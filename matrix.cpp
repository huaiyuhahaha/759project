#include "matrix.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

Matrix::Matrix(int rows, int cols)
    : rows_(rows), cols_(cols), data_(new double[checked_size(rows, cols)]{})
{}

Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), data_(new double[other.size()])
{
    std::copy_n(other.data_, size(), data_);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        Matrix copy(other);
        swap(copy);
    }
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_)
{
    other.rows_ = 0;
    other.cols_ = 0;
    other.data_ = nullptr;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        delete[] data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
        other.rows_ = 0;
        other.cols_ = 0;
        other.data_ = nullptr;
    }
    return *this;
}

Matrix::~Matrix() { delete[] data_; }

int Matrix::rows() const noexcept { return rows_; }
int Matrix::cols() const noexcept { return cols_; }

std::size_t Matrix::size() const noexcept {
    return static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);
}

void Matrix::swap(Matrix& other) noexcept {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(data_, other.data_);
}

double* Matrix::operator[](int row) {
    check_row(row);
    return data_ + static_cast<std::size_t>(row) * cols_;
}

const double* Matrix::operator[](int row) const {
    check_row(row);
    return data_ + static_cast<std::size_t>(row) * cols_;
}

Matrix Matrix::operator+(const Matrix& other) const {
    require_same_shape(other, "addition");
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < size(); ++i)
        result.data_[i] = data_[i] + other.data_[i];
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    require_same_shape(other, "subtraction");
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < size(); ++i)
        result.data_[i] = data_[i] - other.data_[i];
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_)
        throw std::invalid_argument("Matrix multiplication dimensions do not match");

    Matrix result(rows_, other.cols_);
    // i-k-j order reads the right-hand row and writes the result row contiguously.
    for (int i = 0; i < rows_; ++i) {
        for (int k = 0; k < cols_; ++k) {
            const double a = (*this)[i][k];
            for (int j = 0; j < other.cols_; ++j)
                result[i][j] += a * other[k][j];
        }
    }
    return result;
}

std::size_t Matrix::checked_size(int rows, int cols) {
    if (rows < 0 || cols < 0)
        throw std::invalid_argument("Matrix dimensions must be nonnegative");
    const auto r = static_cast<std::size_t>(rows);
    const auto c = static_cast<std::size_t>(cols);
    if (c != 0 && r > std::numeric_limits<std::size_t>::max() / c)
        throw std::overflow_error("Matrix size is too large");
    return r * c;
}

void Matrix::check_row(int row) const {
    if (row < 0 || row >= rows_)
        throw std::out_of_range("Matrix row index out of range");
}

void Matrix::require_same_shape(const Matrix& other, const char* operation) const {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument(std::string("Matrix ") + operation +
                                    " requires matching dimensions");
}

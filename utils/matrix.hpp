#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <type_traits>
#include <vector>

#include "utils/vector.hpp"

#include "exceptions/vector_matrix_exceptions.hpp"

/**
 * @brief A dense row-major matrix with common linear algebra operations.
 * @tparam T Arithmetic element type, excluding bool.
 */
template <typename T>
class Matrix {
    static_assert(
        std::is_arithmetic_v<std::remove_cv_t<T>> &&
        !std::is_same_v<std::remove_cv_t<T>, bool>,
        "Matrix<T>: T must be a real or integer numeric type, excluding complex/bool"
    );

    size_t rows_, cols_;
    std::vector<T> data_;

    /**
     * @brief Converts a 2D index into a flat row-major index.
     * @param r Row index.
     * @param c Column index.
     * @return Flat index into the storage vector.
     */
    size_t idx(size_t r, size_t c) const { return r * cols_ + c; }

    /**
     * @brief Validates matrix coordinates.
     * @param r Row index.
     * @param c Column index.
     * @throw std::out_of_range If coordinates are outside the matrix bounds.
     */
    void check_bounds(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix index out of bounds");
        }
    }

public:
    /**
     * @brief Constructs a matrix with given dimensions and fill value.
     * @param r Number of rows.
     * @param c Number of columns.
     * @param val Initial value for all elements.
     */
    Matrix(size_t r = 0, size_t c = 0, T val = T{})
        : rows_(r),
          cols_(c),
          data_(r * c, val) {}

    /**
     * @brief Returns the number of rows.
     * @return Row count.
     */
    size_t rows() const noexcept { return rows_; }

    /**
     * @brief Returns the number of columns.
     * @return Column count.
     */
    size_t cols() const noexcept { return cols_; }

    /**
     * @brief Accesses an element with bounds checking.
     * @param r Row index.
     * @param c Column index.
     * @return Constant reference to the element.
     * @throw std::out_of_range If coordinates are outside the matrix bounds.
     */
    const T& at(size_t r, size_t c) const { check_bounds(r, c); return data_[idx(r, c)]; }

    /**
     * @brief Accesses an element with bounds checking.
     * @param r Row index.
     * @param c Column index.
     * @return Mutable reference to the element.
     * @throw std::out_of_range If coordinates are outside the matrix bounds.
     */
    T& at(size_t r, size_t c) { check_bounds(r, c); return data_[idx(r, c)]; }

    /**
     * @brief Builds an identity matrix.
     * @param n Matrix size.
     * @return n-by-n identity matrix.
     */
    static Matrix identity(size_t n) {
        Matrix I(n, n, T{});
        for (size_t i = 0; i < n; ++i) I.at(i, i) = T{1};
        return I;
    }

    Matrix operator+(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw DimensionMismatchError("Matrix addition dimension mismatch");
        }
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) res.data_[i] = data_[i] + other.data_[i];
        return res;
    }

    Matrix operator-(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw DimensionMismatchError("Matrix subtraction dimension mismatch");
        }
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) res.data_[i] = data_[i] - other.data_[i];
        return res;
    }

    Matrix operator*(T scalar) const {
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) res.data_[i] = data_[i] * scalar;
        return res;
    }

    friend Matrix operator*(T scalar, const Matrix& m) {
        return m * scalar;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw DimensionMismatchError("Matrix multiplication dimension mismatch");
        }
        Matrix res(rows_, other.cols_, T{});
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t k = 0; k < cols_; ++k) {
                const T aik = at(i, k);
                // Reuse the current left-hand coefficient across the whole row
                // of the right-hand matrix to reduce repeated lookups.
                for (size_t j = 0; j < other.cols_; ++j) {
                    res.at(i, j) += aik * other.at(k, j);
                }
            }
        }
        return res;
    }

    Vector<T> operator*(const Vector<T>& v) const {
        if (cols_ != v.size()) {
            throw DimensionMismatchError("Matrix-Vector multiplication dimension mismatch");
        }
        Vector<T> res(rows_, T{});
        for (size_t i = 0; i < rows_; ++i) {
            T sum{};
            for (size_t j = 0; j < cols_; ++j) sum += at(i, j) * v[j];
            res[i] = sum;
        }
        return res;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& M) {
        os << "[";
        for (size_t i = 0; i < M.rows_; ++i) {
            if (i != 0) {
                os << " ";
            }
            os << "[";
            for (size_t j = 0; j < M.cols_; ++j) {
                os << M.at(i,j);
                if (j + 1 < M.cols_) {
                    os << ", ";
                }
            }
            os << "]";
            if (i + 1 < M.rows_) {
                os << '\n';
            }
        }
        os << "]";
        return os;
    }

    /**
     * @brief Returns the transposed matrix.
     * @return Transposed matrix.
     */
    Matrix transpose() const {
        Matrix res(cols_, rows_, T{});
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                res.at(j, i) = at(i, j);
            }
        }
        return res;
    }

    /**
     * @brief Checks whether the matrix is symmetric.
     * @param tol Absolute tolerance for element comparison.
     * @return true if the matrix is square and symmetric within tolerance.
     */
    bool is_symmetric(double tol = 1e-12) const {
        if (rows_ != cols_) return false;
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = i + 1; j < cols_; ++j) {
                if (std::abs(at(i, j) - at(j, i)) > static_cast<T>(tol)) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief Checks whether the matrix is positive definite.
     *
     * The implementation uses a Cholesky-style decomposition test:
     * diagonal entries must remain positive while the lower-triangular factors
     * are built incrementally.
     *
     * @param tol Numerical tolerance used for stability checks.
     * @return true if the matrix is symmetric positive definite.
     */
    bool is_positive_definite(double tol = 1e-12) const {
        if (rows_ != cols_) return false;
        if (!is_symmetric(1e-10)) return false;

        const size_t n = rows_;
        using RealType = std::common_type_t<std::remove_cv_t<T>, double>;
        Matrix<RealType> L(n, n, RealType{0});

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double sum = static_cast<double>(at(i, j));
                // Subtract previously computed factor contributions.
                for (size_t k = 0; k < j; ++k) {
                    sum -= L.at(i, k) * L.at(j, k);
                }

                if (i == j) {
                    if (sum <= tol) return false;
                    L.at(i, j) = std::sqrt(sum);
                } else {
                    if (std::abs(L.at(j, j)) <= tol) return false;
                    L.at(i, j) = sum / L.at(j, j);
                }
            }
        }
        return true;
    }

    /**
     * @brief Solves a linear system A x = b using Gauss-Jordan elimination.
     *
     * Partial pivoting is used to improve numerical stability.
     *
     * @param A Coefficient matrix.
     * @param b Right-hand side vector.
     * @return Solution vector x.
     * @throw DimensionMismatchError If A is not square or dimensions do not match.
     * @throw SingularMatrixError If the matrix is singular or nearly singular.
     */
    static Vector<T> solve(Matrix A, Vector<T> b) {
        if (A.rows_ != A.cols_) {
            throw DimensionMismatchError("Matrix must be square for solve");
        }
        if (A.rows_ != b.size()) {
            throw DimensionMismatchError("Matrix rows must match vector size");
        }

        const size_t n = A.rows_;
        const double eps = 1e-12;

        for (size_t col = 0; col < n; ++col) {
            size_t pivot_row = col;
            double max_val = std::abs(static_cast<double>(A.at(col, col)));

            // Select the row with the largest absolute pivot in the current column.
            for (size_t row = col + 1; row < n; ++row) {
                const double candidate = std::abs(static_cast<double>(A.at(row, col)));
                if (candidate > max_val) {
                    max_val = candidate;
                    pivot_row = row;
                }
            }

            if (max_val < eps) {
                throw SingularMatrixError("Zero pivot detected, matrix is singular");
            }

            if (pivot_row != col) {
                for (size_t j = 0; j < n; ++j) std::swap(A.at(col, j), A.at(pivot_row, j));
                std::swap(b[col], b[pivot_row]);
            }

            const T pivot = A.at(col, col);
            // Normalize the pivot row so the pivot becomes 1.
            for (size_t j = 0; j < n; ++j) {
                if (j != col) A.at(col, j) /= pivot;
            }
            b[col] /= pivot;
            A.at(col, col) = T{1};

            // Eliminate the current column from all other rows.
            for (size_t row = 0; row < n; ++row) {
                if (row == col) continue;
                const T factor = A.at(row, col);
                if (factor == T{}) continue;
                for (size_t j = 0; j < n; ++j) {
                    if (j != col) A.at(row, j) -= factor * A.at(col, j);
                }
                b[row] -= factor * b[col];
                A.at(row, col) = T{};
            }
        }
        return b;
    }
};
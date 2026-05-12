#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "utils/matrix.hpp"
#include "utils/vector.hpp"

#include "exceptions/optimization_exceptions.hpp"

/**
 * @brief Computes a step size for numerical differentiation.
 *
 * The step scales with the magnitude of x to improve numerical stability.
 *
 * @param x Current coordinate value
 * @param base_step Base step size
 * @return Adjusted step size
 */
inline double derivative_step(double x, double base_step) {
    return base_step * std::max(1.0, std::abs(x));
}

/**
 * @brief Computes a numerical gradient using central differences.
 *
 * Approximates the gradient of a scalar function f at point x.
 *
 * @tparam Func Callable type: double f(const Vector<double>&)
 * @param f Objective function
 * @param x Point at which gradient is evaluated
 * @param base_step Base step size for finite differences
 * @return Gradient vector
 * @throw ObjectiveEvaluationError
 */
template <typename Func>
Vector<double> numerical_gradient(
    const Func& f,
    const Vector<double>& x,
    double base_step = 1e-6
) {
    const size_t n = x.size();
    Vector<double> g(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        // adaptive step for dimension i
        const double h = derivative_step(x[i], base_step);

        Vector<double> xp = x;
        Vector<double> xm = x;
        xp[i] += h;
        xm[i] -= h;

        try {
            // central difference approximation
            const double fp = f(xp);
            const double fm = f(xm);
            g[i] = (fp - fm) / (2.0 * h);
        } catch (const std::exception& e) {
            throw ObjectiveEvaluationError("Failed to evaluate objective during gradient computation: " + std::string(e.what()));
        }
    }

    return g;
}

/**
 * @brief Computes a numerical Hessian matrix using finite differences.
 *
 * Uses second-order central differences for diagonal terms and
 * mixed partial derivatives for off-diagonal terms.
 *
 * @tparam Func Callable type: double f(const Vector<double>&)
 * @param f Objective function
 * @param x Point at which Hessian is evaluated
 * @param base_step Base step size for finite differences
 * @return Symmetric Hessian matrix
 * @throw ObjectiveEvaluationError
 */
template <typename Func>
Matrix<double> numerical_hessian(
    const Func& f,
    const Vector<double>& x,
    double base_step = 1e-4
) {
    const size_t n = x.size();
    Matrix<double> H(n, n, 0.0);
    const double fx = f(x);

    // precompute step sizes for each dimension
    std::vector<double> h(n);
    for (size_t i = 0; i < n; ++i) {
        h[i] = derivative_step(x[i], base_step);
    }

    for (size_t i = 0; i < n; ++i) {
        // second derivative (diagonal term)
        {
            Vector<double> xp = x;
            Vector<double> xm = x;
            xp[i] += h[i];
            xm[i] -= h[i];
            try {
                const double fp = f(xp);
                const double fm = f(xm);
                H.at(i, i) = (fp - 2.0 * fx + fm) / (h[i] * h[i]);
            } catch (const std::exception& e) {
                throw ObjectiveEvaluationError("Failed to evaluate objective during Hessian computation: " + std::string(e.what()));
            }
        }

        // mixed partial derivatives (i, j)
        for (size_t j = i + 1; j < n; ++j) {
            Vector<double> xpp = x;
            Vector<double> xpm = x;
            Vector<double> xmp = x;
            Vector<double> xmm = x;

            // evaluate four corner points
            xpp[i] += h[i]; xpp[j] += h[j];
            xpm[i] += h[i]; xpm[j] -= h[j];
            xmp[i] -= h[i]; xmp[j] += h[j];
            xmm[i] -= h[i]; xmm[j] -= h[j];

            try {
                const double fpp = f(xpp);
                const double fpm = f(xpm);
                const double fmp = f(xmp);
                const double fmm = f(xmm);

                // central mixed derivative approximation
                const double value = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j]);
                H.at(i, j) = value;
                H.at(j, i) = value;
            } catch (const std::exception& e) {
                throw ObjectiveEvaluationError("Failed to evaluate objective during Hessian computation: " + std::string(e.what()));
            }
        }
    }

    return H;
}
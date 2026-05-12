#pragma once

#include <functional>

#include "utils/vector.hpp"
#include "utils/matrix.hpp"

/// Set of functions used by the Newton optimizer.
struct NewtonFunctionSet {
    /// Objective function to minimize.
    std::function<double(const Vector<double>&)> objective;
};

/// Search domain and global search settings.
struct NewtonSearchConfig {
    /// Lower bound of the search domain for each coordinate.
    /// Must satisfy lower_bound_i < upper_bound_i for every i.
    Vector<double> lower_bound;
    /// Upper bound of the search domain for each coordinate.
    /// Must satisfy lower_bound_i < upper_bound_i for every i.
    Vector<double> upper_bound;
    /// Total number of starting points used for the global search.
    /// Must be at least 2.     
    size_t grid_resolution = 5;
};

/// Numerical parameters of the Newton optimizer.
struct NewtonNumericConfig {
    /// Maximum number of iterations allowed for a single start.
    /// Must be at least 1.
    size_t max_iter = 50;
    /// Gradient norm threshold for convergence.
    /// Must be positive.
    double grad_tol = 1e-6;
    /// Step-length threshold for convergence.
    /// Must be positive.
    double step_tol = 1e-8;
    /// Threshold for classifying a point as stationary.
    /// Must be positive.
    double stationarity_tol = 1e-6;
    /// Threshold for treating two points as identical.
    /// Must be positive.
    double duplicate_tol = 1e-4;
    /// Armijo condition coefficient.
    /// Must lie in (0, 1).
    double armijo_c1 = 1e-4;
    /// Step reduction factor used in backtracking line search.
    /// Must lie in (0, 1).
    double backtracking_beta = 0.5;
    /// Minimum admissible step size during line search.
    /// Must be positive.
    double min_alpha = 1e-12;
    /// Initial diagonal regularization added to the Hessian.
    /// Must be positive.
    double initial_regularization = 1e-8;
    /// Multiplicative growth factor for regularization.
    /// Must be greater than 1.
    double regularization_growth = 10.0;
    /// Upper bound on Hessian regularization.
    /// Must be positive.
    double max_regularization = 1e8;
    /// Finite-difference step for numerical gradient evaluation.
    /// Must be positive.
    double gradient_step = 1e-6;
    /// Finite-difference step for numerical Hessian evaluation.
    /// Must be positive.
    double hessian_step = 1e-4;
};

/// Full configuration of the Newton optimizer.
struct NewtonOptimizerConfig {
    /// Problem definition.
    NewtonFunctionSet problem;
    /// Search-domain settings.
    NewtonSearchConfig search;
    /// Numerical settings.
    NewtonNumericConfig numeric;
};
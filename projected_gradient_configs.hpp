#pragma once

#include <functional>

#include "utils/vector.hpp"

/**
 * @brief Configuration structures for the projected gradient optimizer
 */
struct ProjectedGradientFunctionSet {
    /// @brief Objective function to minimize
    std::function<double(const Vector<double>&)> objective;
};

/**
 * @brief Structure representing a linear constraint
 */
struct LinearConstraint {
    /// @brief Coefficients of the linear constraint (a1, a2, ..., an)
    Vector<double> coefficients;
    /// @brief Right-hand side value of the constraint
    double rhs{};
    /// @brief Indicates if the constraint is of the form "greater than or equal to" (true) or "less than or equal to" (false)
    bool greater_equal{false};
};

/**
 * @brief Structure representing the domain configuration for the projected gradient optimizer
 */
struct ProjectedGradientDomainConfig {
    /// @brief Lower bounds for each variable
    Vector<double> lower_bound;
    /// @brief Upper bounds for each variable
    Vector<double> upper_bound;
};

/**
 * @brief Structure representing the numeric configuration for the projected gradient optimizer
 */
struct ProjectedGradientNumericConfig {
    /// @brief Maximum number of iterations before termination
    size_t max_iter = 50;
    /// @brief Tolerance for the norm of the gradient to consider convergence
    double grad_tol = 1e-6;
    /// @brief Tolerance for the change in variable values to consider convergence
    double step_tol = 1e-8;
    /// @brief Tolerance for the norm of the projected gradient to consider stationarity
    double stationarity_tol = 1e-6;
    /// @brief Tolerance for considering two points as duplicates (for tracking distinct minima)
    double duplicate_tol = 1e-4;
    /// @brief Armijo condition constant for backtracking line search
    double armijo_c1 = 1e-4;
    /// @brief Backtracking line search reduction factor
    double backtracking_beta = 0.5;
    /// @brief Minimum step size allowed in line search
    double min_alpha = 1e-12;
    /// @brief Initial step size for line search
    double initial_alpha = 1.0;
    /// @brief Finite difference step size for numerical gradient approximation
    size_t grid_resolution = 5;
    /// @brief Finite difference step size for numerical gradient approximation
    double gradient_step = 1e-6;
    /// @brief Finite difference step size for numerical hessian approximation
    double hessian_step = 1e-4;
    /// @brief Tolerance for projection convergence
    double projection_tol{1e-10};
    /// @brief Maximum iterations for projection algorithm
    size_t projection_max_iter{1000};
};

/**
 * @brief Structure representing the configuration for the projected gradient optimizer
 */
struct ProjectedGradientOptimizerConfig {
    /// @brief Set of functions defining the optimization problem
    ProjectedGradientFunctionSet problem;
    /// @brief Domain configuration for the optimization problem
    ProjectedGradientDomainConfig domain;
    /// @brief Numeric configuration for the optimization algorithm
    ProjectedGradientNumericConfig numeric;
    /// @brief List of linear constraints for the optimization problem
    std::vector<LinearConstraint> linear_constraints;
};

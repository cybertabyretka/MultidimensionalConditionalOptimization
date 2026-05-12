#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "newton_configs.hpp"
#include "utils/vector.hpp"
#include "utils/derivatives.hpp"

/**
 * @brief Stores a point and the objective value at that point.
 */
struct OptimizedPoint {
    Vector<double> point;
    double value{};
};

/**
 * @brief Result of a single Newton optimization run.
 */
struct NewtonResult {
    Vector<double> point;
    double value{};
    double gradient_norm{};
    bool converged{false};
    size_t iterations{0};
};

/**
 * @brief Newton-based optimizer with optional regularization and batch support.
 *
 * The optimizer can run from a single starting point or from a set of start
 * points, collect stationary points, and classify minima via a Hessian test.
 */
class NewtonOptimizer {
    NewtonOptimizerConfig config_;
    std::function<double(const Vector<double>&)> objective_;

    std::vector<OptimizedPoint> stationary_points_;
    std::vector<OptimizedPoint> minimum_points_;

    /**
     * @brief Checks whether a point already exists in a list within tolerance.
     * @param list List of optimized points.
     * @param x Point to compare.
     * @param tol Duplicate-detection tolerance.
     * @return true if x matches an existing point in list.
     */
    static bool is_duplicate(
        const std::vector<OptimizedPoint>& list,
        const Vector<double>& x,
        double tol
    );

    /**
     * @brief Validates the optimizer configuration.
     * @throw InputOptimizationError If the objective function is missing.
     */
    void validate_config() const;

    /**
     * @brief Computes a Newton direction with diagonal regularization fallback.
     *
     * If the Hessian is not positive definite, the method increases diagonal
     * regularization until a descent direction is found or falls back to -grad.
     *
     * @param x Current iterate.
     * @param g Current gradient.
     * @param logs Enables diagnostic output.
     * @return A descent direction.
     */
    Vector<double> regularized_newton_direction(
        const Vector<double>& x,
        const Vector<double>& g,
        bool logs
    ) const;

    /**
     * @brief Runs Newton optimization from a single starting point.
     * @param start Starting point.
     * @param logs Enables diagnostic output.
     * @return Optimization result.
     * @throw DimensionMismatchError If the start point is invalid.
     * @throw ConvergenceError If line search fails.
     * @throw NumericalError If the iteration diverges.
     */
    NewtonResult solve_from_start(
        const Vector<double>& start,
        bool logs
    ) const;

    /**
     * @brief Tests whether a point is a local minimum using the Hessian.
     * @param x Point to classify.
     * @return true if the Hessian is positive definite at x.
     */
    bool is_minimum_point(const Vector<double>& x) const;

public:
    /**
     * @brief Constructs an optimizer from a configuration object.
     * @param config Optimizer configuration.
     */
    explicit NewtonOptimizer(NewtonOptimizerConfig config);

    /**
     * @brief Returns all detected stationary points.
     * @return Constant reference to the stationary point list.
     */
    const std::vector<OptimizedPoint>& get_stationary_points() const;

    /**
     * @brief Returns all detected minimum points.
     * @return Constant reference to the minimum point list.
     */
    const std::vector<OptimizedPoint>& get_minimum_points() const;

    /**
     * @brief Removes all stored optimization results.
     */
    void clear_results();

    /**
     * @brief Optimizes from a single starting point.
     * @param start_point Initial point.
     * @param logs Enables diagnostic output.
     * @return Final optimization result.
     */
    NewtonResult optimize(const Vector<double>& start_point, bool logs = false) const;

    /**
     * @brief Optimizes from multiple starting points and stores unique results.
     * @param start_points Collection of initial points.
     * @param logs Enables diagnostic output.
     */
    void optimize(const std::vector<Vector<double>>& start_points, bool logs = false);
};
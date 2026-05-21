#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "projected_gradient_configs.hpp"

#include "utils/vector.hpp"
#include "utils/derivatives.hpp"

/**
 * @brief Structure representing an optimized point
 */
struct OptimizedPoint {
    /// @brief Point in the variable space
    Vector<double> point;
    /// @brief Objective function value at the point
    double value{};
};

/**
 * @brief Structure representing the result of the projected gradient optimization
 */
struct ProjectedGradientResult {
    /// @brief Point in the variable space
    Vector<double> point;
    /// @brief Objective function value at the point
    double value{};
    /// @brief Measure of stationarity at the point
    double stationarity_measure{};
    /// @brief Indicates if the optimization converged
    bool converged{false};
    /// @brief Number of iterations performed
    size_t iterations{0};
};

/**
 * @brief Class implementing the projected gradient optimization algorithm
 */
class ProjectedGradientOptimizer {
    /// @brief Configuration for the optimizer
    ProjectedGradientOptimizerConfig config_;
    /// @brief Objective function to minimize
    std::function<double(const Vector<double>&)> objective_;

    /// @brief List of stationary points found during optimization
    std::vector<OptimizedPoint> stationary_points_;
    /// @brief List of minimum points found during optimization
    std::vector<OptimizedPoint> minimum_points_;

    /**
     * @brief Checks if a point is a duplicate of any point in a list
     * @param list List of points to check against
     * @param x Point to check
     * @param tol Tolerance for considering two points as duplicates
     * @return True if the point is a duplicate, false otherwise
     */
    static bool is_duplicate(
        const std::vector<OptimizedPoint>& list,
        const Vector<double>& x,
        double tol
    );

    /**
     * @brief Validates the optimizer configuration
     * @throw ProjectedGradientConfigError If the configuration is invalid
     * @throw DimensionMismatchError If there are dimension mismatches in the configuration
     */
    void validate_config() const;

    /**
     * @brief Projects a point onto the feasible region defined by the constraints
     * @param x Point to project
     * @return Projected point
     * @throw DimensionMismatchError If the point dimension does not match the constraint dimensions
     * @throw ProjectedGradientOptimizationError If the projection algorithm fails to converge
     */
    Vector<double> project_point(const Vector<double>& x) const;

    /**
     * @brief Checks if a point satisfies the KKT conditions for being a minimum
     * @param x Point to check
     * @return True if the point is a minimum, false otherwise
     */
    bool is_minimum_point(const Vector<double>& x) const;

    /**
     * @brief Solves the optimization problem starting from a given point
     * @param start Starting point
     * @param logs Whether to log the optimization process
     * @return Result of the optimization
     * @throw DimensionMismatchError If the starting point dimension does not match the constraint dimensions
     * @throw NumericalError If there is numerical instability during optimization
     * @throw ProjectedGradientConvegenceError If the optimization fails to converge
     */
    ProjectedGradientResult solve_from_start(
        const Vector<double>& start,
        bool logs
    ) const;

public:
    /** 
     * @brief Constructor for the ProjectedGradientOptimizer
     * @param config Configuration for the optimizer
     */
    explicit ProjectedGradientOptimizer(ProjectedGradientOptimizerConfig config);

    /** 
     * @brief Gets the list of stationary points found during optimization
     * @return Reference to the list of stationary points
     */
    const std::vector<OptimizedPoint>& get_stationary_points() const;
    /** 
     * @brief Gets the list of minimum points found during optimization
     * @return Reference to the list of minimum points
     */
    const std::vector<OptimizedPoint>& get_minimum_points() const;

    /** 
     * @brief Clears the results of the optimization
     */
    void clear_results();

    /** 
     * @brief Performs optimization starting from a single point
     * @param start_point Starting point
     * @param logs Whether to log the optimization process
     * @return Result of the optimization
     */
    ProjectedGradientResult optimize(const Vector<double>& start_point, bool logs = false) const;

    /** 
     * @brief Performs optimization starting from multiple points
     * @param start_points List of starting points
     * @param logs Whether to log the optimization process
     */
    void optimize(const std::vector<Vector<double>>& start_points, bool logs = false);
};

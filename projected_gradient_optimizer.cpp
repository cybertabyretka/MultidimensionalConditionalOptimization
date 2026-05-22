#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>

#include "projected_gradient_configs.hpp"
#include "projected_gradient_optimizer.hpp"

#include "utils/vector.hpp"
#include "utils/matrix.hpp"
#include "utils/derivatives.hpp"

#include "exceptions/numerical_exceptions.hpp"
#include "exceptions/vector_matrix_exceptions.hpp"
#include "exceptions/optimization_exceptions.hpp"
#include "exceptions/intervals_exceptions.hpp"

namespace {

/// @brief Represents a halfspace defined by a normal vector and a right-hand side value.
struct Halfspace {
    Vector<double> normal;
    double rhs{};
};

/// @brief Represents a linear constraint in a form suitable for projection and KKT checks.
struct ConstraintRep {
    Vector<double> normal;
    double rhs{};
};

/**
 * @brief Projects a point onto a halfspace defined by a linear constraint.
 * @param y The point to be projected.
 * @param h The halfspace representing the linear constraint.
 * @return The projection of y onto the halfspace h.
 * @throws InputOptimizationError If the halfspace has an invalid normal vector (zero vector).
 */
Vector<double> project_onto_halfspace(const Vector<double>& y, const Halfspace& h) {
    const double denom = h.normal.dot(h.normal);
    if (denom <= 0.0) {
        throw InputOptimizationError("Invalid linear constraint: zero normal vector");
    }

    const double violation = h.normal.dot(y) - h.rhs;
    if (violation <= 0.0) {
        return y;
    }

    return y - h.normal * (violation / denom);
}

/**
 * @brief Converts a linear constraint into a halfspace representation.
 * @param c The linear constraint to convert.
 * @return The corresponding halfspace representation of the linear constraint.
 */
Halfspace make_halfspace_from_constraint(const LinearConstraint& c) {
    if (c.greater_equal) {
        return Halfspace{c.coefficients * -1.0, -c.rhs};
    }
    return Halfspace{c.coefficients, c.rhs};
}

/**
 * @brief Converts a linear constraint into a representation suitable for KKT checks.
 * @param c The linear constraint to convert.
 * @return The corresponding representation for KKT checks.
 */
ConstraintRep to_constraint_rep(const LinearConstraint& c) {
    if (c.greater_equal) {
        return ConstraintRep{c.coefficients * -1.0, -c.rhs};
    }
    return ConstraintRep{c.coefficients, c.rhs};
}

/**
 * @brief Collects all constraints into a single vector of representations.
 * @param cfg The optimizer configuration.
 * @param dim The dimension of the problem.
 * @return A vector containing all constraint representations.
 */
std::vector<ConstraintRep> collect_constraints(const ProjectedGradientOptimizerConfig& cfg, size_t dim) {
    std::vector<ConstraintRep> constraints;

    if (!cfg.domain.lower_bound.empty()) {
        for (size_t i = 0; i < dim; ++i) {
            Vector<double> upper_normal(dim, 0.0);
            upper_normal[i] = 1.0;
            constraints.push_back(ConstraintRep{upper_normal, cfg.domain.upper_bound[i]});

            Vector<double> lower_normal(dim, 0.0);
            lower_normal[i] = -1.0;
            constraints.push_back(ConstraintRep{lower_normal, -cfg.domain.lower_bound[i]});
        }
    }

    for (const auto& c : cfg.linear_constraints) {
        constraints.push_back(to_constraint_rep(c));
    }

    return constraints;
}

/**
 * @brief Identifies the indices of active constraints at a given point.
 * @param constraints The vector of constraint representations.
 * @param x The point at which to check constraint activity.
 * @param tol The tolerance for determining constraint activity.
 * @return A vector containing the indices of active constraints.
 */
std::vector<size_t> active_constraint_indices(
    const std::vector<ConstraintRep>& constraints,
    const Vector<double>& x,
    double tol
) {
    std::vector<size_t> active;
    for (size_t i = 0; i < constraints.size(); ++i) {
        const double slack = constraints[i].normal.dot(x) - constraints[i].rhs;
        if (std::abs(slack) <= tol) {
            active.push_back(i);
        }
    }
    return active;
}

/**
 * @brief Checks if a point satisfies all constraints within a given tolerance.
 * @param constraints The vector of constraint representations.
 * @param x The point to check.
 * @param tol The tolerance for constraint satisfaction.
 * @return True if the point satisfies all constraints, false otherwise.
 */
bool satisfies_all_constraints(
    const std::vector<ConstraintRep>& constraints,
    const Vector<double>& x,
    double tol
) {
    for (const auto& c : constraints) {
        if (c.normal.dot(x) - c.rhs > tol) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Builds the active constraint matrix from a list of indices.
 * @param constraints The vector of constraint representations.
 * @param indices The indices of active constraints.
 * @param dim The dimension of the problem.
 * @return The active constraint matrix.
 */
Matrix<double> build_active_matrix(
    const std::vector<ConstraintRep>& constraints,
    const std::vector<size_t>& indices,
    size_t dim
) {
    Matrix<double> A(indices.size(), dim, 0.0);
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto& n = constraints[indices[i]].normal;
        for (size_t j = 0; j < dim; ++j) {
            A.at(i, j) = n[j];
        }
    }
    return A;
}

/**
 * @brief Computes a basis for the nullspace of a matrix.
 * @param A The matrix for which to compute the nullspace basis.
 * @param tol The tolerance for determining the rank of the matrix.
 * @return A vector containing the basis vectors for the nullspace.
 */
std::vector<Vector<double>> nullspace_basis(Matrix<double> A, double tol) {
    const size_t m = A.rows();
    const size_t n = A.cols();
    std::vector<int> pivot_row_for_col(n, -1);
    size_t row = 0;

    for (size_t col = 0; col < n && row < m; ++col) {
        size_t pivot = row;
        double best = std::abs(A.at(row, col));
        for (size_t r = row + 1; r < m; ++r) {
            const double cand = std::abs(A.at(r, col));
            if (cand > best) {
                best = cand;
                pivot = r;
            }
        }

        if (best <= tol) {
            continue;
        }

        if (pivot != row) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(A.at(row, j), A.at(pivot, j));
            }
        }

        const double pivot_val = A.at(row, col);
        for (size_t j = 0; j < n; ++j) {
            A.at(row, j) /= pivot_val;
        }

        for (size_t r = 0; r < m; ++r) {
            if (r == row) {
                continue;
            }
            const double factor = A.at(r, col);
            if (std::abs(factor) <= tol) {
                continue;
            }
            for (size_t j = 0; j < n; ++j) {
                A.at(r, j) -= factor * A.at(row, j);
            }
        }

        pivot_row_for_col[col] = static_cast<int>(row);
        ++row;
    }

    std::vector<Vector<double>> basis;
    for (size_t free_col = 0; free_col < n; ++free_col) {
        if (pivot_row_for_col[free_col] != -1) {
            continue;
        }

        Vector<double> z(n, 0.0);
        z[free_col] = 1.0;
        for (size_t col = 0; col < n; ++col) {
            const int pivot_row = pivot_row_for_col[col];
            if (pivot_row >= 0) {
                z[col] = -A.at(static_cast<size_t>(pivot_row), free_col);
            }
        }
        basis.push_back(z);
    }

    return basis;
}

/**
 * @brief Generates all subsets of a given size from a vector of items.
 * @param items The vector of items from which to generate subsets.
 * @param k The size of each subset to generate.
 * @return A vector containing all subsets of the specified size.
 */
std::vector<std::vector<size_t>> generate_subsets(const std::vector<size_t>& items, size_t k) {
    std::vector<std::vector<size_t>> subsets;
    std::vector<size_t> current;

    std::function<void(size_t)> dfs = [&](size_t pos) {
        if (current.size() == k) {
            subsets.push_back(current);
            return;
        }
        if (pos == items.size()) {
            return;
        }

        current.push_back(items[pos]);
        dfs(pos + 1);
        current.pop_back();
        dfs(pos + 1);
    };

    dfs(0);
    return subsets;
}

/**
 * @brief Checks if a point is a KKT minimum.
 * @param objective The objective function.
 * @param config The optimizer configuration.
 * @param x The point to check.
 * @param constraints The vector of constraint representations.
 * @return True if the point is a KKT minimum, false otherwise.
 */
bool kkt_minimum_check(
    const std::function<double(const Vector<double>&)>& objective,
    const ProjectedGradientOptimizerConfig& config,
    const Vector<double>& x,
    const std::vector<ConstraintRep>& constraints
) {
    const double feas_tol = std::max(config.numeric.grad_tol, 1e-8);
    if (!satisfies_all_constraints(constraints, x, feas_tol)) {
        return false;
    }

    const Vector<double> g = numerical_gradient(objective, x, config.numeric.gradient_step);
    const std::vector<size_t> active = active_constraint_indices(constraints, x, feas_tol);

    bool kkt_ok = false;
    if (active.empty()) {
        // Unconstrained case: check if gradient is approximately zero
        kkt_ok = g.norm() <= config.numeric.grad_tol;
    } else {
        const size_t dim = x.size();
        const size_t max_k = std::min(dim, active.size());
        for (size_t k = 1; k <= max_k && !kkt_ok; ++k) {
            for (const auto& subset : generate_subsets(active, k)) {
                Matrix<double> N(dim, k, 0.0);
                for (size_t col = 0; col < k; ++col) {
                    const auto& normal = constraints[subset[col]].normal;
                    for (size_t row = 0; row < dim; ++row) {
                        N.at(row, col) = normal[row];
                    }
                }
                // Solve the KKT system for the current subset of active constraints
                Matrix<double> G = N.transpose() * N;
                Vector<double> rhs = -(N.transpose() * g);

                try {
                    Vector<double> lambda = Matrix<double>::solve(G, rhs);
                    Vector<double> residual = N * lambda + g;
                    if (residual.norm() <= 1e-6) {
                        bool all_nonnegative = true;
                        for (double v : lambda) {
                            if (v < -1e-6) {
                                all_nonnegative = false;
                                break;
                            }
                        }
                        if (all_nonnegative) {
                            kkt_ok = true;
                            break;
                        }
                    }
                } catch (const std::exception&) {
                }
            }
        }
    }
    // If KKT conditions are not satisfied, return false immediately
    if (!kkt_ok) {
        return false;
    }
    // Check second-order sufficient conditions for optimality
    Matrix<double> H = numerical_hessian(objective, x, config.numeric.hessian_step);
    if (active.empty()) {
        return H.is_positive_definite();
    }
    // For active constraints, check positive definiteness on the nullspace of the active constraint normals
    Matrix<double> A = build_active_matrix(constraints, active, x.size());
    std::vector<Vector<double>> basis = nullspace_basis(A, 1e-10);
    if (basis.empty()) {
        return true;
    }
    // Form the matrix of basis vectors and compute the reduced Hessian
    Matrix<double> Z(x.size(), basis.size(), 0.0);
    for (size_t col = 0; col < basis.size(); ++col) {
        for (size_t row = 0; row < x.size(); ++row) {
            Z.at(row, col) = basis[col][row];
        }
    }

    Matrix<double> reduced = Z.transpose() * H * Z;
    return reduced.is_positive_definite();
}

} // namespace

const std::vector<OptimizedPoint>& ProjectedGradientOptimizer::get_stationary_points() const {
    return stationary_points_;
}

const std::vector<OptimizedPoint>& ProjectedGradientOptimizer::get_minimum_points() const {
    return minimum_points_;
}

ProjectedGradientOptimizer::ProjectedGradientOptimizer(ProjectedGradientOptimizerConfig config)
    : config_(std::move(config)),
      objective_(config_.problem.objective) {
    validate_config();
}

void ProjectedGradientOptimizer::clear_results() {
    stationary_points_.clear();
    minimum_points_.clear();
}

bool ProjectedGradientOptimizer::is_duplicate(
    const std::vector<OptimizedPoint>& list,
    const Vector<double>& x,
    double tol
) {
    for (const auto& p : list) {
        if (p.point.equals(x, tol)) {
            return true;
        }
    }
    return false;
}

void ProjectedGradientOptimizer::validate_config() const {
    if (!objective_) {
        throw ProjectedGradientConfigError("Objective must be provided");
    }

    const bool has_box =
        !config_.domain.lower_bound.empty() || !config_.domain.upper_bound.empty();

    if (config_.domain.lower_bound.size() != config_.domain.upper_bound.size()) {
        throw ProjectedGradientConfigError("Search bounds dimension mismatch");
    }

    if (has_box) {
        if (config_.domain.lower_bound.empty()) {
            throw ProjectedGradientConfigError("Both lower and upper bounds must be provided together");
        }
        for (size_t i = 0; i < config_.domain.lower_bound.size(); ++i) {
            if (config_.domain.lower_bound[i] >= config_.domain.upper_bound[i]) {
                throw ProjectedGradientConfigError("Each lower bound must be strictly less than upper bound");
            }
        }
    }

    if (config_.numeric.max_iter == 0) {
        throw ProjectedGradientConfigError("max_iter must be positive");
    }
    if (config_.numeric.grad_tol <= 0.0 ||
        config_.numeric.step_tol <= 0.0 ||
        config_.numeric.stationarity_tol <= 0.0 ||
        config_.numeric.duplicate_tol <= 0.0 ||
        config_.numeric.min_alpha <= 0.0 ||
        config_.numeric.initial_alpha <= 0.0 ||
        config_.numeric.gradient_step <= 0.0 ||
        config_.numeric.hessian_step <= 0.0 ||
        config_.numeric.projection_tol <= 0.0) {
        throw ProjectedGradientConfigError("Numeric tolerances must be positive");
    }

    if (config_.numeric.projection_max_iter == 0) {
        throw ProjectedGradientConfigError("projection_max_iter must be positive");
    }

    if (config_.numeric.armijo_c1 <= 0.0 || config_.numeric.armijo_c1 >= 1.0) {
        throw ProjectedGradientConfigError("armijo_c1 must lie in (0, 1)");
    }
    if (config_.numeric.backtracking_beta <= 0.0 || config_.numeric.backtracking_beta >= 1.0) {
        throw ProjectedGradientConfigError("backtracking_beta must lie in (0, 1)");
    }

    if (config_.numeric.grid_resolution < 2) {
        throw ProjectedGradientConfigError("grid_resolution must be at least 2");
    }

    const size_t dim = !config_.domain.lower_bound.empty()
        ? config_.domain.lower_bound.size()
        : (!config_.linear_constraints.empty()
           ? config_.linear_constraints.front().coefficients.size()
           : 0);

    for (const auto& c : config_.linear_constraints) {
        if (c.coefficients.size() != dim) {
            throw DimensionMismatchError("Linear constraint dimension mismatch");
        }
    }
}

Vector<double> ProjectedGradientOptimizer::project_point(const Vector<double>& x) const {
    const size_t dim = x.size();

    if (!config_.domain.lower_bound.empty() && config_.domain.lower_bound.size() != dim) {
        throw DimensionMismatchError("Point dimension mismatch with lower bounds");
    }
    if (!config_.domain.upper_bound.empty() && config_.domain.upper_bound.size() != dim) {
        throw DimensionMismatchError("Point dimension mismatch with upper bounds");
    }

    std::vector<Halfspace> halfspaces;
    halfspaces.reserve(
        2 * dim + config_.linear_constraints.size()
    );

    if (!config_.domain.lower_bound.empty()) {
        for (size_t i = 0; i < dim; ++i) {
            Vector<double> e(dim, 0.0);
            e[i] = 1.0;

            halfspaces.push_back(Halfspace{e, config_.domain.upper_bound[i]});

            halfspaces.push_back(Halfspace{e * -1.0, -config_.domain.lower_bound[i]});
        }
    }

    for (const auto& c : config_.linear_constraints) {
        if (c.coefficients.size() != dim) {
            throw DimensionMismatchError("Linear constraint dimension mismatch in projection");
        }
        halfspaces.push_back(make_halfspace_from_constraint(c));
    }

    if (halfspaces.empty()) {
        return x;
    }

    Vector<double> y = x;
    std::vector<Vector<double>> corrections(
        halfspaces.size(),
        Vector<double>(dim, 0.0)
    );

    for (size_t outer = 0; outer < config_.numeric.projection_max_iter; ++outer) {
        Vector<double> prev = y;

        for (size_t j = 0; j < halfspaces.size(); ++j) {
            const Vector<double> s = y + corrections[j];
            const Vector<double> p = project_onto_halfspace(s, halfspaces[j]);
            corrections[j] = s - p;
            y = p;
        }

        if ((y - prev).norm() <= config_.numeric.projection_tol) {
            return y;
        }

        for (double v : y) {
            if (std::isnan(v) || std::isinf(v)) {
                throw ProjectedGradientOptimizationError("Projection diverged or overflowed");
            }
        }
    }

    throw ProjectedGradientOptimizationError("Projection onto feasible set did not converge");
}

bool ProjectedGradientOptimizer::is_minimum_point(const Vector<double>& x) const {
    const size_t dim = x.size();
    const std::vector<ConstraintRep> constraints = collect_constraints(config_, dim);
    return kkt_minimum_check(objective_, config_, x, constraints);
}

ProjectedGradientResult ProjectedGradientOptimizer::solve_from_start(
    const Vector<double>& start,
    bool logs
) const {
    validate_config();
    if (start.empty()) {
        throw DimensionMismatchError("Start point is empty");
    }

    auto effective_step_size = [this](double gnorm) {
        return std::min(
            config_.numeric.initial_alpha,
            1.0 / std::max(1.0, gnorm)
        );
    };

    auto projected_stationarity = [this, &effective_step_size](const Vector<double>& point) {
        const Vector<double> g = numerical_gradient(objective_, point, config_.numeric.gradient_step);
        const double tau = effective_step_size(g.norm());
        const Vector<double> step = project_point(point - g * tau) - point;
        return step.norm() / tau;
    };

    Vector<double> x = project_point(start);
    double fx = objective_(x);
    bool converged = false;
    size_t iter = 0;

    if (logs) {
        std::cout << "\n[ProjectedGradient] Start point: " << x
                  << ", f(x0) = " << fx << '\n';
    }

    for (; iter < config_.numeric.max_iter; ++iter) {
        const Vector<double> g = numerical_gradient(objective_, x, config_.numeric.gradient_step);
        const double gnorm = g.norm();

        if (logs) {
            std::cout << "\n[ProjectedGradient] Iteration " << iter << '\n'
                      << "  x_k      = " << x << '\n'
                      << "  f(x_k)   = " << fx << '\n'
                      << "  grad     = " << g << '\n'
                      << "  ||grad|| = " << gnorm << '\n';
        }

        const double current_stationarity = projected_stationarity(x);
        if (current_stationarity <= config_.numeric.stationarity_tol) {
            converged = true;
            if (logs) {
                std::cout << "[ProjectedGradient] Projected stationarity below tolerance.\n"
                          << "  x*      = " << x << '\n'
                          << "  f(x*)   = " << fx << '\n'
                          << "  stationarity = " << current_stationarity << '\n';
            }
            return {x, fx, current_stationarity, converged, iter};
        }

        const double alpha_cap = effective_step_size(gnorm);
        double alpha = alpha_cap;
        bool accepted = false;
        Vector<double> candidate;
        Vector<double> step;
        double fc = fx;
        double stationarity_measure = 0.0;

        while (alpha >= config_.numeric.min_alpha) {
            candidate = project_point(x - g * alpha);
            step = candidate - x;
            stationarity_measure = step.norm() / alpha;

            try {
                fc = objective_(candidate);
            } catch (const std::exception&) {
                if (logs) {
                    std::cout << "  alpha = " << alpha
                              << " -> objective evaluation failed, backtracking.\n";
                }
                alpha *= config_.numeric.backtracking_beta;
                continue;
            }

            const double dir_deriv = g.dot(step);
            const double armijo_rhs = fx + config_.numeric.armijo_c1 * dir_deriv;

            if (logs) {
                std::cout << "  alpha    = " << alpha << '\n'
                          << "  x_trial  = " << candidate << '\n'
                          << "  f(trial) = " << fc << '\n'
                          << "  Armijo RHS = " << armijo_rhs << '\n'
                          << "  stationarity = " << stationarity_measure << '\n';
            }

            if (fc <= armijo_rhs) {
                accepted = true;
                if (logs) {
                    std::cout << "  step accepted\n";
                }
                break;
            }

            if (logs) {
                std::cout << "  step rejected, backtracking.\n";
            }
            alpha *= config_.numeric.backtracking_beta;
        }

        if (!accepted) {
            throw ProjectedGradientConvergenceError("Line search failed to find an admissible projected step");
        }

        if ((candidate - x).norm() < config_.numeric.step_tol) {
            x = candidate;
            fx = fc;
            const double final_stationarity = projected_stationarity(x);
            converged = final_stationarity < config_.numeric.grad_tol;
            if (logs) {
                std::cout << "[ProjectedGradient] Step norm below tolerance.\n"
                          << "  x_next   = " << x << '\n'
                          << "  f(x_next)= " << fx << '\n'
                          << "  stationarity = " << final_stationarity << '\n'
                          << "  converged = " << std::boolalpha << converged << '\n';
            }
            return {x, fx, final_stationarity, converged, iter + 1};
        }

        x = candidate;
        fx = fc;

        if (logs) {
            std::cout << "  x_{k+1} = " << x << '\n'
                      << "  f(x_{k+1}) = " << fx << '\n';
        }

        for (double v : x) {
            if (std::isnan(v) || std::isinf(v)) {
                throw NumericalError("Divergence or overflow in projected gradient iteration");
            }
        }
    }

    const double stationarity_measure = projected_stationarity(x);
    converged = stationarity_measure < config_.numeric.grad_tol;
    if (logs) {
        std::cout << "\n[ProjectedGradient] Max iterations reached.\n"
                  << "  x_final   = " << x << '\n'
                  << "  f_final   = " << fx << '\n'
                  << "  stationarity = " << stationarity_measure << '\n'
                  << "  converged = " << std::boolalpha << converged << '\n';
    }
    return {x, fx, stationarity_measure, converged, iter};
}

ProjectedGradientResult ProjectedGradientOptimizer::optimize(
    const Vector<double>& start_point,
    bool logs
) const {
    return solve_from_start(start_point, logs);
}

void ProjectedGradientOptimizer::optimize(
    const std::vector<Vector<double>>& start_points,
    bool logs
) {
    clear_results();
    if (logs) {
        std::cout << "[ProjectedGradient] Batch optimization started, starts = "
                  << start_points.size() << '\n';
    }
    for (const auto& start : start_points) {
        try {
            ProjectedGradientResult res = solve_from_start(start, logs);
            if (logs) {
                std::cout << "[ProjectedGradient] Result for start " << start << ":\n"
                          << "  point         = " << res.point << '\n'
                          << "  value         = " << res.value << '\n'
                          << "  stationarity  = " << res.stationarity_measure << '\n'
                          << "  converged     = " << std::boolalpha << res.converged << '\n'
                          << "  iterations    = " << res.iterations << '\n';
            }
            if (res.stationarity_measure <= config_.numeric.stationarity_tol) {
                if (!is_duplicate(stationary_points_, res.point, config_.numeric.duplicate_tol)) {
                    stationary_points_.push_back({res.point, res.value});
                    if (logs) {
                        std::cout << "[ProjectedGradient] Stationary point accepted: "
                                  << res.point << ", f = " << res.value << '\n';
                    }
                    if (is_minimum_point(res.point)) {
                        minimum_points_.push_back({res.point, res.value});
                        if (logs) {
                            std::cout << "[ProjectedGradient] Minimum point accepted: "
                                      << res.point << ", f = " << res.value << '\n';
                        }
                    }
                } else if (logs) {
                    std::cout << "[ProjectedGradient] Duplicate stationary point skipped: "
                              << res.point << '\n';
                }
            }
        } catch (const std::exception& e) {
            if (logs) {
                std::cerr << "[ProjectedGradientOptimizer] Start point " << start
                          << " failed: " << e.what() << '\n';
            }
        }
    }
    auto cmp = [](const OptimizedPoint& a, const OptimizedPoint& b) {
        return a.value < b.value;
    };
    std::sort(stationary_points_.begin(), stationary_points_.end(), cmp);
    std::sort(minimum_points_.begin(), minimum_points_.end(), cmp);
    if (logs) {
        std::cout << "[ProjectedGradient] Batch optimization finished.\n"
                  << "  stationary points = " << stationary_points_.size() << '\n'
                  << "  minimum points    = " << minimum_points_.size() << '\n';
    }
}

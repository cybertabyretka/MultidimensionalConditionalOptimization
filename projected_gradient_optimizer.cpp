#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "projected_gradient_configs.hpp"
#include "projected_gradient_optimizer.hpp"

#include "utils/vector.hpp"
#include "utils/matrix.hpp"
#include "utils/derivatives.hpp"

#include "exceptions/numerical_exceptions.hpp"
#include "exceptions/vector_matrix_exceptions.hpp"
#include "exceptions/optimization_exceptions.hpp"
#include "exceptions/intervals_exceptions.hpp"
#include "exceptions/projected_gradient_exceptions.hpp"

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
        throw ProjectedGradientOptimizationError("Objective must be provided");
    }
    if (config_.domain.lower_bound.size() != config_.domain.upper_bound.size()) {
        throw ProjectedGradientConfigError("Search bounds dimension mismatch");
    }
    if (config_.domain.lower_bound.empty()) {
        throw ProjectedGradientConfigError("Search bounds must not be empty");
    }
    for (size_t i = 0; i < config_.domain.lower_bound.size(); ++i) {
        if (config_.domain.lower_bound[i] >= config_.domain.upper_bound[i]) {
            throw ProjectedGradientConfigError("Each lower bound must be strictly less than upper bound");
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
        config_.numeric.hessian_step <= 0.0) {
        throw ProjectedGradientConfigError("Numeric tolerances must be positive");
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
}

Vector<double> ProjectedGradientOptimizer::project_point(const Vector<double>& x) const {
    if (x.size() != config_.domain.lower_bound.size()) {
        throw DimensionMismatchError("Point dimension mismatch with feasible set");
    }

    Vector<double> projected = x;
    for (size_t i = 0; i < x.size(); ++i) {
        if (projected[i] < config_.domain.lower_bound[i]) {
            projected[i] = config_.domain.lower_bound[i];
        }
        if (projected[i] > config_.domain.upper_bound[i]) {
            projected[i] = config_.domain.upper_bound[i];
        }
    }
    return projected;
}

bool ProjectedGradientOptimizer::is_minimum_point(const Vector<double>& x) const {
    Matrix<double> H = numerical_hessian(objective_, x, config_.numeric.hessian_step);
    return H.is_positive_definite();
}

ProjectedGradientResult ProjectedGradientOptimizer::solve_from_start(
    const Vector<double>& start,
    bool logs
) const {
    validate_config();
    if (start.empty()) {
        throw DimensionMismatchError("Start point is empty");
    }

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

        double alpha = config_.numeric.initial_alpha;
        bool accepted = false;
        Vector<double> candidate;
        Vector<double> step;
        double fc = fx;
        double stationarity_measure = 0.0;

        while (alpha >= config_.numeric.min_alpha) {
            candidate = project_point(x - g * alpha);
            step = candidate - x;
            stationarity_measure = step.norm() / alpha;

            if (stationarity_measure < config_.numeric.grad_tol || step.norm() < config_.numeric.step_tol) {
                x = candidate;
                fx = objective_(x);
                converged = true;
                if (logs) {
                    std::cout << "[ProjectedGradient] Projected step below tolerance.\n"
                              << "  x*      = " << x << '\n'
                              << "  f(x*)   = " << fx << '\n'
                              << "  stationarity = " << stationarity_measure << '\n';
                }
                return {x, fx, stationarity_measure, converged, iter + 1};
            }

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
            const Vector<double> gfinal = numerical_gradient(objective_, x, config_.numeric.gradient_step);
            const Vector<double> projected_step = project_point(x - gfinal * config_.numeric.initial_alpha) - x;
            const double final_stationarity = projected_step.norm() / config_.numeric.initial_alpha;
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

    const Vector<double> gfinal = numerical_gradient(objective_, x, config_.numeric.gradient_step);
    const Vector<double> projected_step = project_point(x - gfinal * config_.numeric.initial_alpha) - x;
    const double stationarity_measure = projected_step.norm() / config_.numeric.initial_alpha;
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
            std::cerr << "[ProjectedGradientOptimizer] Start point " << start
                      << " failed: " << e.what() << '\n';
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

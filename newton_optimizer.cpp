#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "newton_configs.hpp"
#include "newton_optimizer.hpp"

#include "utils/vector.hpp"
#include "utils/matrix.hpp"
#include "utils/derivatives.hpp"
#include "utils/grid_generator.hpp"

#include "exceptions/numerical_exceptions.hpp"
#include "exceptions/vector_matrix_exceptions.hpp"
#include "exceptions/optimization_exceptions.hpp"
#include "exceptions/intervals_exceptions.hpp"

const std::vector<OptimizedPoint>& NewtonOptimizer::get_stationary_points() const {
    return stationary_points_;
}

const std::vector<OptimizedPoint>& NewtonOptimizer::get_minimum_points() const {
    return minimum_points_;
}

NewtonOptimizer::NewtonOptimizer(NewtonOptimizerConfig config)
    : config_(std::move(config)),
      objective_(config_.problem.objective) {
    validate_config();
}

bool NewtonOptimizer::is_minimum_point(const Vector<double>& x) const {
    Matrix<double> H = numerical_hessian(objective_, x, config_.numeric.hessian_step);
    return H.is_positive_definite();
}

void NewtonOptimizer::clear_results() {
    stationary_points_.clear();
    minimum_points_.clear();
}

bool NewtonOptimizer::is_duplicate(
    const std::vector<OptimizedPoint>& list,
    const Vector<double>& x,
    double tol
) {
    for (const auto& p : list) {
        if (p.point.equals(x, tol)) return true;
    }
    return false;
}

void NewtonOptimizer::validate_config() const {
    if (!objective_) {
        throw InputOptimizationError("Objective must be provided");
    }
}

Vector<double> NewtonOptimizer::regularized_newton_direction(
    const Vector<double>& x,
    const Vector<double>& g,
    bool logs
) const {
    Matrix<double> H = numerical_hessian(objective_, x, config_.numeric.hessian_step);

    if (H.rows() != H.cols() || H.rows() != x.size()) {
        throw DimensionMismatchError("Hessian dimension mismatch");
    }
    if (logs) {
        std::cout << "[Newton] Hessian at x = " << x << ":\n" << H << '\n';
    }

    // Start with the raw Hessian and progressively shift the diagonal if needed.
    Matrix<double> H_reg = H;
    double lambda = 0.0;

    for (size_t attempt = 0; attempt < 30; ++attempt) {
        if (attempt > 0) {
            lambda = (lambda == 0.0)
                ? config_.numeric.initial_regularization
                : lambda * config_.numeric.regularization_growth;

            if (lambda > config_.numeric.max_regularization) {
                if (logs) {
                    std::cout << "[Newton] Regularization exceeded max_regularization = "
                              << config_.numeric.max_regularization << '\n';
                }
                break;
            }

            H_reg = H;
            for (size_t i = 0; i < H_reg.rows(); ++i) {
                H_reg.at(i, i) += lambda;
            }
            if (logs) {
                std::cout << "[Newton] Regularization attempt " << attempt
                          << ", lambda = " << lambda << '\n';
            }
        }

        if (!H_reg.is_positive_definite()) {
            if (logs) {
                std::cout << "[Newton] H_reg is not positive definite, continue.\n";
            }
            continue;
        }

        // Solve H d = -g. A valid Newton step must also be a descent direction.
        Vector<double> d = Matrix<double>::solve(H_reg, g * -1.0);
        if (logs) {
            std::cout << "[Newton] Candidate Newton direction d = " << d
                      << ", g^T d = " << g.dot(d) << '\n';
        }
        if (g.dot(d) < 0.0) {
            return d;
        }
    }
    // Safe fallback if the regularized Newton step cannot produce descent.
    if (logs) {
        std::cout << "[Newton] Falling back to steepest descent direction d = -g\n";
    }
    return g * -1.0;
}

NewtonResult NewtonOptimizer::solve_from_start(
    const Vector<double>& start,
    bool logs
) const {
    validate_config();
    if (start.empty()) {
        throw DimensionMismatchError("Start point is empty");
    }

    Vector<double> x = start;
    double fx = objective_(x);
    bool converged = false;
    size_t iter = 0;
    if (logs) {
        std::cout << "\n[Newton] Start point: " << x
                  << ", f(x0) = " << fx << '\n';
    }
    for (; iter < config_.numeric.max_iter; ++iter) {
        const Vector<double> g = numerical_gradient(objective_, x, config_.numeric.gradient_step);
        const double gnorm = g.norm();
        if (logs) {
            std::cout << "\n[Newton] Iteration " << iter << '\n'
                      << "  x_k      = " << x << '\n'
                      << "  f(x_k)   = " << fx << '\n'
                      << "  grad     = " << g << '\n'
                      << "  ||grad|| = " << gnorm << '\n';
        }
        if (gnorm < config_.numeric.grad_tol) {
            converged = true;
            fx = objective_(x);

            if (logs) {
                std::cout << "[Newton] Gradient norm below tolerance, stop.\n"
                          << "  x*      = " << x << '\n'
                          << "  f(x*)   = " << fx << '\n'
                          << "  ||grad||= " << gnorm << '\n';
            }
            return {x, fx, gnorm, converged, iter};
        }

        Vector<double> d = regularized_newton_direction(x, g, logs);
        double dir_deriv = g.dot(d);

        if (dir_deriv >= 0.0) {
            // Enforce descent explicitly if the Newton direction is numerically unsafe.
            if (logs) {
                std::cout << "[Newton] Direction is not descent, fallback to -grad.\n";
            }
            d = g * -1.0;
            dir_deriv = -gnorm * gnorm;
        }
        if (logs) {
            std::cout << "  direction = " << d << '\n'
                      << "  g^T d     = " << dir_deriv << '\n';
        }
        double alpha = 1.0;
        bool accepted = false;
        Vector<double> candidate;
        double fc = fx;

        while (alpha >= config_.numeric.min_alpha) {
            candidate = x + d * alpha;
            try {
                fc = objective_(candidate);
            } catch (const std::exception&) {
                // Objective evaluation may fail for invalid trial points; backtrack.
                if (logs) {
                    std::cout << "  alpha = " << alpha
                              << " -> objective evaluation failed, backtracking.\n";
                }
                alpha *= config_.numeric.backtracking_beta;
                continue;
            }

            const double armijo_rhs =
                fx + config_.numeric.armijo_c1 * alpha * dir_deriv;
            if (logs) {
                std::cout << "  alpha    = " << alpha << '\n'
                          << "  x_trial  = " << candidate << '\n'
                          << "  f(trial) = " << fc << '\n'
                          << "  Armijo RHS = " << armijo_rhs << '\n';
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
            throw ConvergenceError("Line search failed to find a decreasing step");
        }
        if ((candidate - x).norm() < config_.numeric.step_tol) {
            x = candidate;
            fx = fc;
            const double final_gnorm = numerical_gradient(
                objective_, x, config_.numeric.gradient_step
            ).norm();
            converged = final_gnorm < config_.numeric.grad_tol;
            if (logs) {
                std::cout << "[Newton] Step norm below tolerance.\n"
                          << "  x_next   = " << x << '\n'
                          << "  f(x_next)= " << fx << '\n'
                          << "  ||grad|| = " << final_gnorm << '\n'
                          << "  converged = " << std::boolalpha << converged << '\n';
            }
            return {x, fx, final_gnorm, converged, iter + 1};
        }
        x = candidate;
        fx = fc;
        if (logs) {
            std::cout << "  x_{k+1} = " << x << '\n'
                      << "  f(x_{k+1}) = " << fx << '\n';
        }
        // Protect against numerical blow-up before the next iteration.
        for (double v : x) {
            if (std::isnan(v) || std::isinf(v)) {
                throw NumericalError("Divergence or overflow in Newton iteration");
            }
        }
    }
    const Vector<double> gfinal = numerical_gradient(objective_, x, config_.numeric.gradient_step);
    const double gnorm = gfinal.norm();
    converged = gnorm < config_.numeric.grad_tol;
    if (logs) {
        std::cout << "\n[Newton] Max iterations reached.\n"
                  << "  x_final   = " << x << '\n'
                  << "  f_final   = " << fx << '\n'
                  << "  grad      = " << gfinal << '\n'
                  << "  ||grad||  = " << gnorm << '\n'
                  << "  converged = " << std::boolalpha << converged << '\n';
    }
    return {x, fx, gnorm, converged, iter};
}

NewtonResult NewtonOptimizer::optimize(
    const Vector<double>& start_point,
    bool logs
) const {
    return solve_from_start(start_point, logs);
}

void NewtonOptimizer::optimize(
    const std::vector<Vector<double>>& start_points,
    bool logs
) {
    clear_results();
    if (logs) {
        std::cout << "[Newton] Batch optimization started, starts = "
                  << start_points.size() << '\n';
    }
    for (const auto& start : start_points) {
        try {
            NewtonResult res = solve_from_start(start, logs);
            if (logs) {
                std::cout << "[Newton] Result for start " << start << ":\n"
                          << "  point         = " << res.point << '\n'
                          << "  value         = " << res.value << '\n'
                          << "  grad_norm     = " << res.gradient_norm << '\n'
                          << "  converged     = " << std::boolalpha << res.converged << '\n'
                          << "  iterations    = " << res.iterations << '\n';
            }
            if (res.gradient_norm <= config_.numeric.stationarity_tol) {
                if (!is_duplicate(stationary_points_, res.point, config_.numeric.duplicate_tol)) {
                    stationary_points_.push_back({res.point, res.value});
                    if (logs) {
                        std::cout << "[Newton] Stationary point accepted: "
                                  << res.point << ", f = " << res.value << '\n';
                    }
                    if (is_minimum_point(res.point)) {
                        minimum_points_.push_back({res.point, res.value});
                        if (logs) {
                            std::cout << "[Newton] Minimum point accepted: "
                                      << res.point << ", f = " << res.value << '\n';
                        }
                    }
                } else if (logs) {
                    std::cout << "[Newton] Duplicate stationary point skipped: "
                              << res.point << '\n';
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[NewtonOptimizer] Start point " << start
                      << " failed: " << e.what() << '\n';
        }
    }
    auto cmp = [](const OptimizedPoint& a, const OptimizedPoint& b) {
        return a.value < b.value;
    };
    std::sort(stationary_points_.begin(), stationary_points_.end(), cmp);
    std::sort(minimum_points_.begin(), minimum_points_.end(), cmp);
    if (logs) {
        std::cout << "[Newton] Batch optimization finished.\n"
                  << "  stationary points = " << stationary_points_.size() << '\n'
                  << "  minimum points    = " << minimum_points_.size() << '\n';
    }
}
#pragma once

#include <cmath>
#include <initializer_list>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "projected_gradient_optimizer.hpp"

#include "utils/vector.hpp"
#include "utils/matrix.hpp"
#include "utils/derivatives.hpp"

#include "tests/utils.hpp"

#include "exceptions/optimization_exceptions.hpp"
#include "exceptions/vector_matrix_exceptions.hpp"

inline LinearConstraint make_constraint(
    std::initializer_list<double> coeffs,
    double rhs,
    bool greater_equal = false
) {
    LinearConstraint c;
    c.coefficients = Vector<double>(coeffs);
    c.rhs = rhs;
    c.greater_equal = greater_equal;
    return c;
}

template <typename Objective>
inline ProjectedGradientOptimizerConfig make_optimizer_config(
    Objective objective,
    const std::vector<LinearConstraint>& constraints
) {
    ProjectedGradientOptimizerConfig cfg;
    cfg.problem.objective = objective;
    cfg.linear_constraints = constraints;
    cfg.numeric.max_iter = 200;
    cfg.numeric.grad_tol = 1e-6;
    cfg.numeric.step_tol = 1e-8;
    cfg.numeric.stationarity_tol = 1e-6;
    cfg.numeric.duplicate_tol = 1e-4;
    cfg.numeric.armijo_c1 = 1e-4;
    cfg.numeric.backtracking_beta = 0.5;
    cfg.numeric.min_alpha = 1e-12;
    cfg.numeric.initial_alpha = 1.0;
    cfg.numeric.gradient_step = 1e-6;
    cfg.numeric.hessian_step = 1e-4;
    cfg.numeric.projection_tol = 1e-10;
    cfg.numeric.projection_max_iter = 1000;
    return cfg;
}

/**
 * @brief Test the projected gradient optimizer on a simple quadratic function with linear constraints
 */
inline void test_optimizer_quadratic() {
    try {
        // f(x) = (x1 - 2)^2 + 3 * (x2 - 1)^2
        auto f = [](const Vector<double>& x) -> double {
            return (x[0] - 2.0) * (x[0] - 2.0) + 3.0 * (x[1] - 1.0) * (x[1] - 1.0);
        };
        // Constraints: x1 >= 0, x2 >= 0, x1 + x2 <= 5
        std::vector<LinearConstraint> constraints = {
            make_constraint({1.0, 0.0}, 0.0, true),
            make_constraint({0.0, 1.0}, 0.0, true),
            make_constraint({1.0, 1.0}, 5.0, false)
        };
        ProjectedGradientOptimizer optimizer(make_optimizer_config(f, constraints));
        ProjectedGradientResult result = optimizer.optimize(Vector<double>{-4.0, 7.0}, false);
        if (!result.converged) {
            throw std::logic_error("Optimizer did not converge for constrained quadratic");
        }
        if (!result.point.equals(Vector<double>{2.0, 1.0}, 1e-3)) {
            throw std::logic_error("Constrained quadratic solution is incorrect");
        }
        if (std::abs(result.value) > 1e-6) {
            throw std::logic_error("Constrained quadratic value is incorrect");
        }
        try {
            // Invalid configuration: mismatched constraint dimension
            std::vector<LinearConstraint> bad_constraints = {
                make_constraint({1.0, 1.0, 1.0}, 1.0, false)
            };
            ProjectedGradientOptimizer bad_optimizer(make_optimizer_config(f, bad_constraints));
            bad_optimizer.optimize(Vector<double>{0.0, 0.0}, false);
            throw std::logic_error("Invalid configuration should have failed");
        } catch (const DimensionMismatchError&) {}
    } catch (const std::logic_error& e) {
        print_test_failed("Optimizer_Quadratic", e.what());
    }
}

/**
 * @brief Test the projected gradient optimizer on an anisotropic quadratic function with multiple linear constraints
 */
inline void test_optimizer_anisotropic_quadratic() {
    try {
        // f(x) = (x1 - 10)^2 + 100 * (x2 - 10)^2
        auto f = [](const Vector<double>& x) -> double {
            const double x1 = x[0];
            const double x2 = x[1];
            return (x1 - 10.0) * (x1 - 10.0) + 100.0 * (x2 - 10.0) * (x2 - 10.0);
        };
        // Constraints: 3x1 - x2 <= 12, 2x1 + 5x2 <= 30, 3x1 + 2x2 <= 22, x1 - 3x2 <= 0
        //              2x1 + 5x2 >= 10, 5x1 + x2 >= 5, x1 >= 0, x2 >= 0
        std::vector<LinearConstraint> constraints = {
            make_constraint({-1.0, 3.0}, 12.0, false),
            make_constraint({2.0, 5.0}, 30.0, false),
            make_constraint({3.0, 2.0}, 22.0, false),
            make_constraint({1.0, -3.0}, 0.0, false),
            make_constraint({2.0, 5.0}, 10.0, true),
            make_constraint({5.0, 1.0}, 5.0, true),
            make_constraint({1.0, 0.0}, 0.0, true),
            make_constraint({0.0, 1.0}, 0.0, true)
        };
        ProjectedGradientOptimizer optimizer(make_optimizer_config(f, constraints));
        std::vector<Vector<double>> starts = {
            Vector<double>{0.0, 0.0},
            Vector<double>{2.0, 5.0},
            Vector<double>{5.0, 5.0}
        };
        optimizer.optimize(starts, false);
        const auto& minimum_points = optimizer.get_minimum_points();
        if (minimum_points.empty()) {
            throw std::logic_error("Anisotropic quadratic minimum was not found");
        }
        const Vector<double> expected{2.73, 4.91};
        if (!minimum_points[0].point.equals(expected, 0.05)) {
            throw std::logic_error("Anisotropic quadratic minimum coordinates are incorrect");
        }
        if (minimum_points[0].value > 2650.0) {
            throw std::logic_error("Anisotropic quadratic minimum value is too large");
        }
    } catch (const std::logic_error& e) {
        print_test_failed("Optimizer_AnisotropicQuadratic", e.what());
    }
}

/**
 * @brief Test the projected gradient optimizer on a simple linear function with linear constraints
 */
inline void test_optimizer_linear() {
    try {
        // f(x) = x1 + 2*x2
        auto f = [](const Vector<double>& x) -> double {
            return x[0] + 2.0 * x[1];
        };
        // Constraints: x1 >= 0, x2 >= 0, x1 + x2 <= 1
        std::vector<LinearConstraint> constraints = {
            make_constraint({1.0, 0.0}, 0.0, true),
            make_constraint({0.0, 1.0}, 0.0, true),
            make_constraint({1.0, 1.0}, 1.0, false)
        };
        ProjectedGradientOptimizer optimizer(make_optimizer_config(f, constraints));
        std::vector<Vector<double>> starts = {
            Vector<double>{0.4, 0.4},
            Vector<double>{0.9, 0.1},
            Vector<double>{0.1, 0.9}
        };
        optimizer.optimize(starts, false);
        const auto& minimum_points = optimizer.get_minimum_points();
        if (minimum_points.empty()) {
            throw std::logic_error("No minimum points found for boundary constrained problem");
        }
        if (!minimum_points[0].point.equals(Vector<double>{0.0, 0.0}, 1e-3)) {
            throw std::logic_error("Boundary minimum should be at the origin");
        }
        if (std::abs(minimum_points[0].value) > 1e-6) {
            throw std::logic_error("Boundary minimum value is incorrect");
        }
    } catch (const std::logic_error& e) {
        print_test_failed("Optimizer_Linear", e.what());
    }
}

/**
 * @brief Test the projected gradient optimizer on a saddle point problem with linear constraints
 */
inline void test_optimizer_saddle_point() {
    try {
        // f(x) = -x1^2 + x2^2
        auto f = [](const Vector<double>& x) -> double {
            return -x[0] * x[0] + x[1] * x[1];
        };
        // Constraints: x1 >= 1, x1 <= -1, x2 >= 0, x2 <= 0
        //               x1 + x2 >= 0, x1 + x2 <= 0
        std::vector<LinearConstraint> constraints = {
            make_constraint({1.0, 0.0}, 1.0, false),
            make_constraint({1.0, 0.0}, -1.0, true),
            make_constraint({0.0, 1.0}, 0.0, false),
            make_constraint({0.0, 1.0}, 0.0, true)
        };
        ProjectedGradientOptimizer optimizer(make_optimizer_config(f, constraints));
        std::vector<Vector<double>> starts = {
            Vector<double>{0.0, 0.0},
            Vector<double>{0.5, 0.0},
            Vector<double>{-0.5, 0.0}
        };
        optimizer.optimize(starts, false);
        const auto& stationary_points = optimizer.get_stationary_points();
        const auto& minimum_points = optimizer.get_minimum_points();
        if (stationary_points.empty()) {
            throw std::logic_error("Expected a stationary point at the origin");
        }
        // The origin is a saddle point, so it should be classified as stationary but not minimum
        bool origin_is_stationary = false;
        for (const auto& p : stationary_points) {
            if (p.point.equals(Vector<double>{0.0, 0.0}, 1e-6)) {
                origin_is_stationary = true;
                break;
            }
        }
        if (!origin_is_stationary) {
            throw std::logic_error("Origin should be stationary for the constrained saddle test");
        }
        for (const auto& p : minimum_points) {
            if (p.point.equals(Vector<double>{0.0, 0.0}, 1e-6)) {
                throw std::logic_error("Origin must not be classified as a minimum");
            }
        }
    } catch (const std::logic_error& e) {
        print_test_failed("Optimizer_SaddlePoint", e.what());
    }
}

/**
 * @brief Test the projected gradient optimizer on a function with multiple minima within the constraints
 */
inline void test_optimizer_multiple_minima() {
    try {
        // f(x) = x1^2 * (x1 - 2)^2 + x2^2 * (x2 - 2)^2
        auto f = [](const Vector<double>& x) -> double {
            const double a = x[0] * x[0] * (x[0] - 2.0) * (x[0] - 2.0);
            const double b = x[1] * x[1] * (x[1] - 2.0) * (x[1] - 2.0);
            return a + b;
        };
        // Constraints: 0 <= x1 <= 2, 0 <= x2 <= 2
        std::vector<LinearConstraint> constraints = {
            make_constraint({1.0, 0.0}, 0.0, true),
            make_constraint({0.0, 1.0}, 0.0, true),
            make_constraint({1.0, 0.0}, 2.0, false),
            make_constraint({0.0, 1.0}, 2.0, false)
        };
        ProjectedGradientOptimizer optimizer(make_optimizer_config(f, constraints));
        std::vector<Vector<double>> starts = {
            Vector<double>{0.0, 0.0},
            Vector<double>{0.0, 2.0},
            Vector<double>{2.0, 0.0},
            Vector<double>{2.0, 2.0},
            Vector<double>{1.0, 1.0}
        };
        optimizer.optimize(starts, false);
        const auto& minimum_points = optimizer.get_minimum_points();
        if (minimum_points.size() < 4) {
            throw std::logic_error("Expected four distinct minima on the box corners");
        }
        const std::vector<Vector<double>> expected = {
            Vector<double>{0.0, 0.0},
            Vector<double>{0.0, 2.0},
            Vector<double>{2.0, 0.0},
            Vector<double>{2.0, 2.0}
        };
        for (const auto& target : expected) {
            bool found = false;
            for (const auto& p : minimum_points) {
                if (p.point.equals(target, 1e-3)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::logic_error("Missing expected corner minimum");
            }
        }
    } catch (const std::logic_error& e) {
        print_test_failed("Optimizer_MultipleMinima", e.what());
    }
}

/**
 * @brief Numerical gradient - normal, edge and exceptional cases
 */
void test_numerical_gradient() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            return x[0] * x[0] + 3.0 * x[1] * x[1];
        };
        // Standard case
        {
            Vector<double> x{1.5, -2.0};
            Vector<double> g = numerical_gradient(f, x, 1e-6);
            Vector<double> expected{3.0, -12.0};

            if (!g.equals(expected, 1e-4))
                throw std::logic_error("Numerical gradient incorrect for quadratic function");
        }
        // Extreme case: very small values
        {
            Vector<double> x{1e-12, -1e-12};
            Vector<double> g = numerical_gradient(f, x, 1e-6);
            Vector<double> expected{2e-12, -6e-12};
            if (!g.equals(expected, 1e-8))
                throw std::logic_error("Numerical gradient inaccurate near zero");
        }
        // Exceptional case: objective throws an exception
        {
            auto throwing_f = [](const Vector<double>& x) -> double {
                if (x[0] > 0.4) {
                    throw ObjectiveEvaluationError("objective failed in gradient test");
                }
                return x[0] * x[0] + x[1] * x[1];
            };
            try {
                (void)numerical_gradient(throwing_f, Vector<double>{0.5, 0.0}, 1e-6);
                throw std::logic_error("numerical_gradient should have propagated exception");
            } catch (const ObjectiveEvaluationError&) {}
        }
    } catch (const std::logic_error& e) {
        print_test_failed("Numerical_Gradient", e.what());
    }
}

/**
 * @brief Numerical hessian - normal, edge and exceptional cases
 */
void test_numerical_hessian() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            return x[0] * x[0] + 3.0 * x[0] * x[1] + 2.0 * x[1] * x[1];
        };
        // Standard case
        {
            Vector<double> x{1.0, -1.0};
            Matrix<double> H = numerical_hessian(f, x, 1e-4);
            Matrix<double> expected(2, 2, 0.0);
            expected.at(0, 0) = 2.0;
            expected.at(0, 1) = 3.0;
            expected.at(1, 0) = 3.0;
            expected.at(1, 1) = 4.0;
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    if (!double_equals(H.at(i, j), expected.at(i, j), 1e-3))
                        throw std::logic_error("Numerical hessian incorrect for quadratic function");
                }
            }
        }
        // Extreme case: flat function near zero
        {
            auto flat_f = [](const Vector<double>& x) -> double {
                return x[0] * x[0] * x[0] * x[0] + x[1] * x[1] * x[1] * x[1];
            };
            Matrix<double> H = numerical_hessian(flat_f, Vector<double>{0.0, 0.0}, 1e-4);
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    if (std::abs(H.at(i, j)) > 1e-6)
                        throw std::logic_error("Numerical hessian should be near zero at the flat point");
                }
            }
        }
        // Exceptional case: objective throws an exception
        {
            auto throwing_f = [](const Vector<double>& x) -> double {
                if (x[0] > 0.1) {
                    throw ObjectiveEvaluationError("objective failed in hessian test");
                }
                return x[0] * x[0] + x[1] * x[1];
            };
            try {
                (void)numerical_hessian(throwing_f, Vector<double>{0.2, 0.0}, 1e-4);
                throw std::logic_error("numerical_hessian should have propagated exception");
            } catch (const ObjectiveEvaluationError&) {}
        }
    } catch (const std::logic_error& e) {
        print_test_failed("Numerical_Hessian", e.what());
    }
}
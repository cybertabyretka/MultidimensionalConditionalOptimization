#pragma once

#include <cmath>
#include <numbers>
#include <stdexcept>

#include "newton_optimizer.hpp"
#include "newton_configs.hpp"

#include "utils/vector.hpp"
#include "utils/matrix.hpp"
#include "utils/derivatives.hpp"

#include "tests/utils.hpp"

#include "exceptions/intervals_exceptions.hpp"
#include "exceptions/optimization_exceptions.hpp"

/**
 * @brief Optimizer - Simple quadratic function (standard case)
 * 
 * Function: f(x) = x1^2 + x2^2, minimum at (0, 0), value 0
 */
void test_optimizer_quadratic() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            return x[0] * x[0] + x[1] * x[1];
        };
        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.search.lower_bound = Vector<double>{-5.0, -5.0};
        cfg.search.upper_bound = Vector<double>{5.0, 5.0};
        cfg.search.grid_resolution = 3;
        cfg.numeric.max_iter = 50;
        cfg.numeric.grad_tol = 1e-6;
        cfg.numeric.gradient_step = 1e-6;
        cfg.numeric.hessian_step = 1e-4;
        NewtonOptimizer optimizer(cfg);
        // Run from one starting point
        NewtonResult result = optimizer.optimize(Vector<double>{2.0, 3.0}, false);
        // Check convergence
        if (!result.converged)
            throw std::logic_error("Optimizer did not converge for simple quadratic");
        // Check that the found point is close to (0, 0)
        if (!result.point.equals(Vector<double>{0.0, 0.0}, 1e-4))
            throw std::logic_error("Solution not close to (0,0)");
        // Check the function value at the found point
        double f_at_solution = f(result.point);
        if (!double_equals(f_at_solution, 0.0, 1e-6))
            throw std::logic_error("Function value at solution not close to 0");
    } catch (const std::exception& e) {
        print_test_failed("Optimizer_Quadratic", e.what());
    }
}

/**
 * @brief Optimizer - Rosenbrock function (hard standard case)
 * 
 * Classic function: f(x1,x2) = (1-x1)^2 + 100*(x2-x1^2)^2.
 * Minimum at (1, 1), value 0.
 */
void test_optimizer_rosenbrock() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            const double x1 = x[0], x2 = x[1];
            return (1.0 - x1) * (1.0 - x1) + 100.0 * (x2 - x1 * x1) * (x2 - x1 * x1);
        };
        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.search.lower_bound = Vector<double>{-2.0, -2.0};
        cfg.search.upper_bound = Vector<double>{2.0, 2.0};
        cfg.search.grid_resolution = 5;
        cfg.numeric.max_iter = 100;
        cfg.numeric.grad_tol = 1e-6;
        cfg.numeric.gradient_step = 1e-6;
        cfg.numeric.hessian_step = 1e-4;
        NewtonOptimizer optimizer(cfg);
        std::vector<Vector<double>> starts;
        starts.push_back(Vector<double>{-1.0, -1.0});
        starts.push_back(Vector<double>{0.0, 0.0});
        starts.push_back(Vector<double>{1.5, 1.5});
        optimizer.optimize(starts, false);
        // Check that the optimizer found at least one point
        const auto& min_points = optimizer.get_minimum_points();
        if (min_points.empty())
            throw std::logic_error("No minimum points found for Rosenbrock");
        // Check that the best point is close to (1, 1)
        const auto& best = min_points[0];
        if (!best.point.equals(Vector<double>{1.0, 1.0}, 5e-2))
            throw std::logic_error("Best solution not close to (1, 1)");
        // Check that the function value is acceptable
        if (best.value > 1e-3)
            throw std::logic_error("Function value at best point too high");
    } catch (const std::exception& e) {
        print_test_failed("Optimizer_Rosenbrock", e.what());
    }
}

/**
 * @brief Optimizer - Saddle point function (extreme case)
 * 
 * Function: f(x1,x2) = (x1^2-1)^2 + (x2^2-1)^2 + 0.5*x1*x2. 
 * Minima are near (±1, ±1), saddle points are elsewhere
 */
void test_optimizer_saddle_point() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            const double x1 = x[0], x2 = x[1];
            const double term1 = (x1 * x1 - 1.0) * (x1 * x1 - 1.0);
            const double term2 = (x2 * x2 - 1.0) * (x2 * x2 - 1.0);
            return term1 + term2 + 0.5 * x1 * x2;
        };
        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.search.lower_bound = Vector<double>{-2.0, -2.0};
        cfg.search.upper_bound = Vector<double>{2.0, 2.0};
        cfg.search.grid_resolution = 5;
        cfg.numeric.max_iter = 50;
        cfg.numeric.grad_tol = 1e-6;
        cfg.numeric.stationarity_tol = 1e-6;
        cfg.numeric.gradient_step = 1e-6;
        cfg.numeric.hessian_step = 1e-4;
        NewtonOptimizer optimizer(cfg);
        std::vector<Vector<double>> starts;
        starts.push_back(Vector<double>{0.5, 0.5});
        starts.push_back(Vector<double>{1.5, 1.5});
        starts.push_back(Vector<double>{-1.5, 1.5});
        optimizer.optimize(starts, false);
        // Check that minimum points were found
        const auto& min_points = optimizer.get_minimum_points();
        if (min_points.empty())
            throw std::logic_error("No minimum points found");
        // Check that the best point has an acceptable value (close to the minima)
        // Minima should be close to (±1, ±1)
        const auto& best = min_points[0];
        const bool x1_valid = (best.point[0] > 0.5 && best.point[0] < 1.5) ||
                              (best.point[0] < -0.5 && best.point[0] > -1.5);
        const bool x2_valid = (best.point[1] > 0.5 && best.point[1] < 1.5) ||
                              (best.point[1] < -0.5 && best.point[1] > -1.5);
        if (!(x1_valid && x2_valid))
            throw std::logic_error("Best solution not in expected region");
    } catch (const std::exception& e) {
        print_test_failed("Optimizer_SaddlePoint", e.what());
    }
}

/**
 * @brief Optimizer - Multiple local minima (extreme case)
 * 
 * Function: f(x) = sin(pi*x1) * sin(pi*x2), multiple local minima and maxima
 */
void test_optimizer_multiple_minima() {
    try {
        const double PI = std::numbers::pi;
        
        auto f = [PI](const Vector<double>& x) -> double {
            return std::sin(PI * x[0]) * std::sin(PI * x[1]);
        };
        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.search.lower_bound = Vector<double>{-2.0, -2.0};
        cfg.search.upper_bound = Vector<double>{2.0, 2.0};
        cfg.search.grid_resolution = 7;
        cfg.numeric.max_iter = 50;
        cfg.numeric.grad_tol = 1e-6;
        cfg.numeric.gradient_step = 1e-6;
        cfg.numeric.hessian_step = 1e-4;
        NewtonOptimizer optimizer(cfg);
        std::vector<Vector<double>> starts;
        // Different starting points
        starts.push_back(Vector<double>{0.3, 0.3});
        starts.push_back(Vector<double>{-0.3, 0.3});
        starts.push_back(Vector<double>{1.2, 1.2});
        optimizer.optimize(starts, false);
        const auto& min_points = optimizer.get_minimum_points();
        // Check that several distinct minima were found, or at least one
        if (min_points.empty())
            throw std::logic_error("No minimum points found");
        if (min_points[0].value > -0.5)
            throw std::logic_error("Minimum point should have negative value");
    } catch (const std::exception& e) {
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
    } catch (const std::exception& e) {
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
    } catch (const std::exception& e) {
        print_test_failed("Numerical_Hessian", e.what());
    }
}
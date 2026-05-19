#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "projected_gradient_configs.hpp"
#include "projected_gradient_optimizer.hpp"

#include "utils/vector.hpp"

using Vec = Vector<double>;

auto objective = [](const Vec& x) -> double {
    const double x1 = x[0];
    const double x2 = x[1];
    return (x1 - 10.0) * (x1 - 10.0) + 100.0 * (x2 - 10.0) * (x2 - 10.0);
};

LinearConstraint make_constraint(std::initializer_list<double> coeffs, double rhs, bool ge = false) {
    LinearConstraint c;
    c.coefficients = Vector<double>(coeffs);
    c.rhs = rhs;
    c.greater_equal = ge;
    return c;
}

ProjectedGradientOptimizerConfig make_config() {
    ProjectedGradientOptimizerConfig cfg;
    cfg.problem.objective = objective;
    cfg.linear_constraints = {
        make_constraint({-1.0, 3.0}, 12.0, false),
        make_constraint({2.0, 5.0}, 30.0, false),
        make_constraint({3.0, 2.0}, 22.0, false),
        make_constraint({1.0, -3.0}, 0.0, false),
        make_constraint({2.0, 5.0}, 10.0, true),
        make_constraint({5.0, 1.0}, 5.0, true),
        make_constraint({1.0, 0.0}, 0.0, true),
        make_constraint({0.0, 1.0}, 0.0, true)
    };
    cfg.numeric.max_iter = 200;
    cfg.numeric.grad_tol = 1e-6;
    cfg.numeric.step_tol = 1e-10;
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

int main() {
    std::cout << std::fixed << std::setprecision(10);

    ProjectedGradientOptimizer optimizer(make_config());
    std::vector<Vec> starts = {
        Vec{0.0, 0.0},
        Vec{1.0, 2.0},
        Vec{2.0, 5.0},
        Vec{3.0, 3.0},
        Vec{5.0, 5.0},
        Vec{8.0, 1.0},
        Vec{1.0, 8.0},
        Vec{10.0, 10.0}
    };

    optimizer.optimize(starts, true);

    std::cout << "\nStationary points:\n";
    for (const auto& p : optimizer.get_stationary_points()) {
        std::cout << p.point << " -> f = " << p.value << '\n';
    }

    std::cout << "\nMinimum points:\n";
    for (const auto& p : optimizer.get_minimum_points()) {
        std::cout << p.point << " -> f = " << p.value << '\n';
    }

    return 0;
}

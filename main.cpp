#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "newton_configs.hpp"
#include "newton_optimizer.hpp"

#include "utils/vector.hpp"
#include "utils/matrix.hpp"

using Vec = Vector<double>;

auto f = [](const Vec& x) -> double {
    const double x1 = x[0];
    const double x2 = x[1];
    return std::pow(x2 - x1 * x1, 2) + 100.0 * std::pow(1.0 - x1, 2);
};

NewtonOptimizerConfig make_config() {
    NewtonOptimizerConfig cfg;
    cfg.problem.objective = f;
    cfg.numeric.max_iter = 5;
    cfg.numeric.grad_tol = 1e-6;
    cfg.numeric.step_tol = 1e-10;
    cfg.numeric.stationarity_tol = 1e-6;
    return cfg;
}

void run_from_start(const Vec& start, bool logs = true) {
    NewtonOptimizer optimizer(make_config());

    std::cout << "\n==================================================\n";
    std::cout << "Start point: " << start << '\n';
    std::cout << "==================================================\n";

    [[maybe_unused]] auto result = optimizer.optimize(start, logs);
}

void first_test() {
    std::cout << "\n==================== small_numbers ====================\n";

    const std::vector<Vec> starts = {
        Vec{-1.5,  -1.5},
        Vec{ 2.5,   2.5},
        Vec{ 0.0,   0.0},
    };

    for (const auto& s : starts) {
        run_from_start(s, true);
    }
}

void second_test() {
    std::cout << "\n==================== big_numbers ====================\n";

    const std::vector<Vec> starts = {
        Vec{-20.0, -20.0},
        Vec{-25.0,  25.0},
    };

    for (const auto& s : starts) {
        run_from_start(s, true);
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(10);

    first_test();
    second_test();

    return 0;
}
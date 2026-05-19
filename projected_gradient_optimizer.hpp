#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "projected_gradient_configs.hpp"

#include "utils/vector.hpp"
#include "utils/derivatives.hpp"

struct OptimizedPoint {
    Vector<double> point;
    double value{};
};

struct ProjectedGradientResult {
    Vector<double> point;
    double value{};
    double stationarity_measure{};
    bool converged{false};
    size_t iterations{0};
};

class ProjectedGradientOptimizer {
    ProjectedGradientOptimizerConfig config_;
    std::function<double(const Vector<double>&)> objective_;

    std::vector<OptimizedPoint> stationary_points_;
    std::vector<OptimizedPoint> minimum_points_;

    static bool is_duplicate(
        const std::vector<OptimizedPoint>& list,
        const Vector<double>& x,
        double tol
    );

    void validate_config() const;

    Vector<double> project_point(const Vector<double>& x) const;

    bool is_minimum_point(const Vector<double>& x) const;

    ProjectedGradientResult solve_from_start(
        const Vector<double>& start,
        bool logs
    ) const;

public:
    explicit ProjectedGradientOptimizer(ProjectedGradientOptimizerConfig config);

    const std::vector<OptimizedPoint>& get_stationary_points() const;
    const std::vector<OptimizedPoint>& get_minimum_points() const;

    void clear_results();

    ProjectedGradientResult optimize(const Vector<double>& start_point, bool logs = false) const;

    void optimize(const std::vector<Vector<double>>& start_points, bool logs = false);
};

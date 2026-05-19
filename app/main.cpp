#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "projected_gradient_optimizer.hpp"

#include "app/config_parser.hpp"
#include "app/conditions_parser.hpp"
#include "app/function_parser.hpp"

#include "utils/grid_generator.hpp"
#include "utils/vector.hpp"

#include "exceptions/config_exceptions.hpp"
#include "exceptions/latex_parser_exceptions.hpp"
#include "exceptions/optimization_exceptions.hpp"
#include "exceptions/vector_matrix_exceptions.hpp"

inline auto make_objective(
    const std::string& expr,
    const std::vector<std::string>& vars
) {
    auto parsed = parse_function(expr, vars);

    return [parsed](const Vector<double>& x) -> double {
        std::vector<double> vals(x.begin(), x.end());
        return parsed(vals);
    };
}

inline std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw ConfigFileOpenException(path);
    }

    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

inline void print_points(const std::vector<OptimizedPoint>& points) {
    std::cout << std::fixed << std::setprecision(10);

    for (const auto& p : points) {
        std::cout << p.point << " -> f = " << p.value << '\n';
    }
}

inline size_t infer_domain_dimension(
    const ProjectedGradientDomainConfig& cfg,
    const std::vector<LinearConstraint>& linear_constraints
) {
    if (!cfg.lower_bound.empty()) {
        return cfg.lower_bound.size();
    }
    if (!linear_constraints.empty()) {
        return linear_constraints.front().coefficients.size();
    }
    return 0;
}

inline std::vector<Vector<double>> generate_fallback_starts(size_t dim) {
    std::vector<Vector<double>> starts;
    starts.reserve(2 * dim + 1);

    Vector<double> zero(dim);
    for (size_t i = 0; i < dim; ++i) {
        zero[i] = 0.0;
    }
    starts.push_back(zero);

    for (size_t i = 0; i < dim; ++i) {
        Vector<double> p(dim);
        for (size_t j = 0; j < dim; ++j) {
            p[j] = 0.0;
        }

        p[i] = 1.0;
        starts.push_back(p);

        p[i] = -1.0;
        starts.push_back(p);
    }

    return starts;
}

int run_app(
    const std::string& command,
    const std::string& config_path,
    const std::string& objective_path,
    const std::string& conditions_path,
    bool log_enabled
) {
    const std::string objective_text = read_text_file(objective_path);
    const std::vector<std::string> variables = extract_variables_in_order(objective_text);

    if (variables.empty()) {
        throw LaTeXParserException("No variables found in objective file");
    }

    ProjectedGradientDomainConfig domain_cfg = parse_conditions_from_xml(conditions_path);
    ProjectedGradientNumericConfig numeric_cfg = parse_numeric_config_from_xml(config_path);

    const std::string conditions_text = read_text_file(conditions_path);
    std::vector<LinearConstraint> linear_constraints = parse_linear_constraints_from_xml(conditions_text);

    const size_t domain_dim = infer_domain_dimension(domain_cfg, linear_constraints);
    if (domain_dim == 0) {
        throw InputOptimizationError("Failed to infer domain dimension from conditions");
    }

    if (domain_dim != variables.size()) {
        throw DimensionMismatchError(
            "conditions have " + std::to_string(domain_dim) +
            " variables, but objective contains " + std::to_string(variables.size())
        );
    }

    ProjectedGradientOptimizerConfig cfg;
    cfg.domain = domain_cfg;
    cfg.numeric = numeric_cfg;
    cfg.linear_constraints = std::move(linear_constraints);
    cfg.problem.objective = make_objective(objective_text, variables);

    std::vector<Vector<double>> starts;
    if (!cfg.domain.lower_bound.empty()) {
        GridGenerator grid;
        starts = grid.generate(
            cfg.domain.lower_bound,
            cfg.domain.upper_bound,
            cfg.numeric.grid_resolution
        );
    } else {
        starts = generate_fallback_starts(domain_dim);
    }

    ProjectedGradientOptimizer optimizer(cfg);
    optimizer.optimize(starts, log_enabled);

    if (command == "find_min") {
        print_points(optimizer.get_minimum_points());
        return 0;
    }

    if (command == "find_stat") {
        print_points(optimizer.get_stationary_points());
        return 0;
    }

    throw InputOptimizationError("Unknown command: " + command);
}

inline void print_usage(const char* program_name) {
    std::cerr
        << "Usage:\n"
        << "  " << program_name << " find_min  <config.xml> <objective.txt> <conditions.xml> [--log|--logs]\n"
        << "  " << program_name << " find_stat <config.xml> <objective.txt> <conditions.xml> [--log|--logs]\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 5 && argc != 6) {
            print_usage(argv[0]);
            return 1;
        }

        const std::string command = argv[1];
        const std::string config_path = argv[2];
        const std::string objective_path = argv[3];
        const std::string conditions_path = argv[4];

        bool log_enabled = false;
        if (argc == 6) {
            const std::string flag = argv[5];
            if (flag != "--log" && flag != "--logs") {
                print_usage(argv[0]);
                return 1;
            }
            log_enabled = true;
        }

        return run_app(command, config_path, objective_path, conditions_path, log_enabled);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

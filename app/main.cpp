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
        throw std::runtime_error("Cannot open file: " + path);
    }

    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

inline void print_points(const std::vector<OptimizedPoint>& points) {
    std::cout << std::fixed << std::setprecision(10);

    for (const auto& p : points) {
        std::cout << p.point << " -> f = " << p.value << '\n';
    }
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
        throw std::runtime_error("No variables found in objective file");
    }

    ProjectedGradientDomainConfig domain_cfg = parse_conditions_from_xml(conditions_path);
    ProjectedGradientNumericConfig numeric_cfg = parse_numeric_config_from_xml(config_path);

    if (domain_cfg.lower_bound.size() != variables.size()) {
        throw std::runtime_error(
            "Dimension mismatch: conditions have " + std::to_string(domain_cfg.lower_bound.size()) +
            " variables, but objective contains " + std::to_string(variables.size())
        );
    }

    ProjectedGradientOptimizerConfig cfg;
    cfg.domain = domain_cfg;
    cfg.numeric = numeric_cfg;
    cfg.problem.objective = make_objective(objective_text, variables);

    GridGenerator grid;
    std::vector<Vector<double>> starts = grid.generate(
        cfg.domain.lower_bound,
        cfg.domain.upper_bound,
        cfg.numeric.grid_resolution
    );

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

    throw std::runtime_error("Unknown command: " + command);
}

inline void print_usage(const char* program_name) {
    std::cerr
        << "Usage:\n"
        << "  " << program_name << " find_min  <config.xml> <objective.txt> <conditions.xml> [--log]\n"
        << "  " << program_name << " find_stat <config.xml> <objective.txt> <conditions.xml> [--log]\n";
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
            if (std::string(argv[5]) != "--log") {
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

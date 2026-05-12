#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "newton_optimizer.hpp"

#include "app/config_parser.hpp"
#include "app/function_parser.hpp"

#include "utils/grid_generator.hpp"
#include "utils/vector.hpp"

auto make_objective(
    const std::string& expr,
    const std::vector<std::string>& vars
) {
    auto parsed = parse_function(expr, vars);

    return [parsed](const Vector<double>& x) -> double {
        std::vector<double> vals(x.begin(), x.end());
        return parsed(vals);
    };
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

std::vector<std::string> extract_variables_in_order(const std::string& expr) {
    std::vector<std::string> vars;
    const auto terms = parse_polynomial(expr);
    for (const auto& term : terms) {
        for (const auto& [var, power] : term.powers) {
            (void)power;
            if (std::find(vars.begin(), vars.end(), var) == vars.end()) {
                vars.push_back(var);
            }
        }
    }
    return vars;
}

void print_points(const std::vector<OptimizedPoint>& points) {
    std::cout << std::fixed << std::setprecision(10);

    for (const auto& p : points) {
        std::cout << p.point << " -> f = " << p.value << '\n';
    }
}

int run_app(const std::string& command,
            const std::string& config_path,
            const std::string& expr_path,
            bool log_enabled) {
    const std::string expr_text = read_text_file(expr_path);
    const std::vector<std::string> variables = extract_variables_in_order(expr_text);

    if (variables.empty()) {
        throw std::runtime_error("No variables found in expression file");
    }

    NewtonSearchConfig search_cfg = load_newton_search_config_from_xml(config_path);
    NewtonNumericConfig numeric_cfg = load_newton_numeric_config_from_xml(config_path);

    if (search_cfg.lower_bound.size() != variables.size()) {
        throw std::runtime_error(
            "Dimension mismatch: config has " + std::to_string(search_cfg.lower_bound.size()) +
            " variables, but expression contains " + std::to_string(variables.size())
        );
    }

    NewtonOptimizerConfig cfg;
    cfg.search = search_cfg;
    cfg.numeric = numeric_cfg;
    cfg.problem.objective = make_objective(expr_text, variables);

    GridGenerator grid;
    std::vector<Vector<double>> starts = grid.generate(
        cfg.search.lower_bound,
        cfg.search.upper_bound,
        cfg.search.grid_resolution
    );

    NewtonOptimizer optimizer(cfg);
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

void print_usage(const char* program_name) {
    std::cerr
        << "Usage:\n"
        << "  " << program_name << " find_min  <config.xml> <expression.txt> [--log]\n"
        << "  " << program_name << " find_stat <config.xml> <expression.txt> [--log]\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 4 && argc != 5) {
            print_usage(argv[0]);
            return 1;
        }

        const std::string command = argv[1];
        const std::string config_path = argv[2];
        const std::string expr_path = argv[3];

        bool log_enabled = false;
        if (argc == 5) {
            if (std::string(argv[4]) != "--log") {
                print_usage(argv[0]);
                return 1;
            }
            log_enabled = true;
        }

        return run_app(command, config_path, expr_path, log_enabled);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
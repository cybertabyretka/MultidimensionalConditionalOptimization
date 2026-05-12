#pragma once

#include <string>

#include "app/parser_utils.hpp"
#include "exceptions/config_exceptions.hpp"
#include "projected_gradient_configs.hpp"
#include "exceptions/projected_gradient_exceptions.hpp"

inline void set_default_projected_gradient_numeric_config(ProjectedGradientNumericConfig& cfg) {
    cfg.max_iter = 50;
    cfg.grad_tol = 1e-6;
    cfg.step_tol = 1e-8;
    cfg.stationarity_tol = 1e-6;
    cfg.duplicate_tol = 1e-4;
    cfg.armijo_c1 = 1e-4;
    cfg.backtracking_beta = 0.5;
    cfg.min_alpha = 1e-12;
    cfg.initial_alpha = 1.0;
    cfg.grid_resolution = 5;
    cfg.gradient_step = 1e-6;
    cfg.hessian_step = 1e-4;
}

inline ProjectedGradientNumericConfig parse_numeric_config_from_xml(const std::string& filename) {
    ProjectedGradientNumericConfig cfg;
    set_default_projected_gradient_numeric_config(cfg);

    std::string xml = read_file_to_string(filename);

    if (!get_tag_size_t(xml, "max_iter", cfg.max_iter)) {
        throw ConfigParseException("Failed to read max_iter from config");
    }
    if (!get_tag_double(xml, "grad_tol", cfg.grad_tol)) {
        throw ConfigParseException("Failed to read grad_tol from config");
    }
    if (!get_tag_double(xml, "step_tol", cfg.step_tol)) {
        throw ConfigParseException("Failed to read step_tol from config");
    }
    if (!get_tag_double(xml, "stationarity_tol", cfg.stationarity_tol)) {
        throw ConfigParseException("Failed to read stationarity_tol from config");
    }
    if (!get_tag_double(xml, "duplicate_tol", cfg.duplicate_tol)) {
        throw ConfigParseException("Failed to read duplicate_tol from config");
    }
    if (!get_tag_double(xml, "armijo_c1", cfg.armijo_c1)) {
        throw ConfigParseException("Failed to read armijo_c1 from config");
    }
    if (!get_tag_double(xml, "backtracking_beta", cfg.backtracking_beta)) {
        throw ConfigParseException("Failed to read backtracking_beta from config");
    }
    if (!get_tag_double(xml, "min_alpha", cfg.min_alpha)) {
        throw ConfigParseException("Failed to read min_alpha from config");
    }
    if (!get_tag_double(xml, "initial_alpha", cfg.initial_alpha)) {
        throw ConfigParseException("Failed to read initial_alpha from config");
    }
    if (!get_tag_size_t(xml, "grid_resolution", cfg.grid_resolution)) {
        throw ConfigParseException("Failed to read grid_resolution from config");
    }
    if (!get_tag_double(xml, "gradient_step", cfg.gradient_step)) {
        throw ConfigParseException("Failed to read gradient_step from config");
    }
    if (!get_tag_double(xml, "hessian_step", cfg.hessian_step)) {
        throw ConfigParseException("Failed to read hessian_step from config");
    }

    if (cfg.max_iter == 0) {
        throw ProjectedGradientConfigError("max_iter must be positive");
    }
    if (cfg.grad_tol <= 0.0 || cfg.step_tol <= 0.0 || cfg.stationarity_tol <= 0.0 || cfg.duplicate_tol <= 0.0) {
        throw ProjectedGradientConfigError("Tolerances must be positive");
    }
    if (cfg.armijo_c1 <= 0.0 || cfg.armijo_c1 >= 1.0) {
        throw ProjectedGradientConfigError("armijo_c1 must lie in (0, 1)");
    }
    if (cfg.backtracking_beta <= 0.0 || cfg.backtracking_beta >= 1.0) {
        throw ProjectedGradientConfigError("backtracking_beta must lie in (0, 1)");
    }
    if (cfg.min_alpha <= 0.0 || cfg.initial_alpha <= 0.0) {
        throw ProjectedGradientConfigError("alpha parameters must be positive");
    }
    if (cfg.grid_resolution < 2) {
        throw ProjectedGradientConfigError("grid_resolution must be at least 2");
    }
    if (cfg.gradient_step <= 0.0 || cfg.hessian_step <= 0.0) {
        throw ProjectedGradientConfigError("Finite difference steps must be positive");
    }

    return cfg;
}

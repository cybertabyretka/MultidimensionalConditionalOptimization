#pragma once

#include <string>

#include "app/parser_utils.hpp"
#include "exceptions/config_exceptions.hpp"
#include "projected_gradient_configs.hpp"
#include "exceptions/projected_gradient_exceptions.hpp"

inline void set_default_conditions_config(ProjectedGradientDomainConfig& cfg) {
    cfg.lower_bound = Vector<double>();
    cfg.upper_bound = Vector<double>();
}

inline ProjectedGradientDomainConfig parse_conditions_from_xml(const std::string& filename) {
    ProjectedGradientDomainConfig cfg;
    set_default_conditions_config(cfg);

    std::string xml = read_file_to_string(filename);

    if (!get_tag_vector_double(xml, "lower_bound", cfg.lower_bound)) {
        throw ConfigParseException("Failed to read lower_bound from conditions file");
    }

    if (!get_tag_vector_double(xml, "upper_bound", cfg.upper_bound)) {
        throw ConfigParseException("Failed to read upper_bound from conditions file");
    }

    if (cfg.lower_bound.size() != cfg.upper_bound.size()) {
        throw ProjectedGradientConfigError("lower_bound and upper_bound must have the same dimension");
    }

    if (cfg.lower_bound.empty()) {
        throw ProjectedGradientConfigError("lower_bound and upper_bound must not be empty");
    }

    for (size_t i = 0; i < cfg.lower_bound.size(); ++i) {
        if (cfg.lower_bound[i] >= cfg.upper_bound[i]) {
            throw ProjectedGradientConfigError("Each lower_bound element must be strictly less than upper_bound");
        }
    }

    return cfg;
}

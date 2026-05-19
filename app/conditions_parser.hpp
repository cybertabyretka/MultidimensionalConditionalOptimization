#pragma once

#include <string>
#include <vector>

#include "app/parser_utils.hpp"

#include "exceptions/config_exceptions.hpp"

#include "projected_gradient_configs.hpp"

#include "exceptions/optimization_exceptions.hpp"

inline void set_default_conditions_config(ProjectedGradientDomainConfig& cfg) {
    cfg.lower_bound = Vector<double>();
    cfg.upper_bound = Vector<double>();
}

inline std::string decode_xml_entities(std::string s) {
    for (size_t pos = 0; (pos = s.find("&lt;", pos)) != std::string::npos; ) {
        s.replace(pos, 4, "<");
        pos += 1;
    }
    for (size_t pos = 0; (pos = s.find("&gt;", pos)) != std::string::npos; ) {
        s.replace(pos, 4, ">");
        pos += 1;
    }
    for (size_t pos = 0; (pos = s.find("&amp;", pos)) != std::string::npos; ) {
        s.replace(pos, 5, "&");
        pos += 1;
    }
    return s;
}

inline bool parse_constraint_sense(const std::string& raw, bool& greater_equal) {
    const std::string s = decode_xml_entities(trim_copy(raw));

    if (s == "<=" || s == "le" || s == "LE") {
        greater_equal = false;
        return true;
    }
    if (s == ">=" || s == "ge" || s == "GE") {
        greater_equal = true;
        return true;
    }
    return false;
}

inline std::vector<LinearConstraint> parse_linear_constraints_from_xml(const std::string& xml) {
    std::vector<LinearConstraint> constraints;

    std::string constraints_block;
    if (!get_tag_string(xml, "constraints", constraints_block)) {
        return constraints;
    }

    const std::string open_tag = "<constraint>";
    const std::string close_tag = "</constraint>";

    size_t pos = 0;
    while (true) {
        size_t start = constraints_block.find(open_tag, pos);
        if (start == std::string::npos) {
            break;
        }

        start += open_tag.size();
        size_t end = constraints_block.find(close_tag, start);
        if (end == std::string::npos) {
            throw ConfigParseException("Unclosed <constraint> block in conditions file");
        }

        std::string block = constraints_block.substr(start, end - start);

        LinearConstraint c;
        if (!get_tag_vector_double(block, "coefficients", c.coefficients)) {
            throw ConfigParseException("Failed to read <coefficients> in constraint");
        }
        if (!get_tag_double(block, "rhs", c.rhs)) {
            throw ConfigParseException("Failed to read <rhs> in constraint");
        }

        std::string sense;
        if (!get_tag_string(block, "sense", sense)) {
            throw ConfigParseException("Failed to read <sense> in constraint");
        }
        if (!parse_constraint_sense(sense, c.greater_equal)) {
            throw ConfigParseException("Constraint sense must be <= or >=");
        }

        constraints.push_back(std::move(c));
        pos = end + close_tag.size();
    }

    return constraints;
}

inline ProjectedGradientDomainConfig parse_conditions_from_xml(const std::string& filename) {
    ProjectedGradientDomainConfig cfg;
    set_default_conditions_config(cfg);

    std::string xml = read_file_to_string(filename);

    const bool has_lower = get_tag_vector_double(xml, "lower_bound", cfg.lower_bound);
    const bool has_upper = get_tag_vector_double(xml, "upper_bound", cfg.upper_bound);

    if (has_lower != has_upper) {
        throw ProjectedGradientConfigError("lower_bound and upper_bound must be provided together");
    }

    if (has_lower) {
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
    }

    const std::vector<LinearConstraint> linear_constraints = parse_linear_constraints_from_xml(xml);

    if (!has_lower && linear_constraints.empty()) {
        throw ProjectedGradientConfigError(
            "Conditions file must contain either box bounds or linear constraints"
        );
    }

    return cfg;
}

#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <cctype>
#include <algorithm>

#include "exceptions/config_exceptions.hpp"

#include "newton_configs.hpp"

std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };

    auto it1 = std::find_if(s.begin(), s.end(), not_space);
    auto it2 = std::find_if(s.rbegin(), s.rend(), not_space).base();

    if (it1 >= it2) return {};
    return std::string(it1, it2);
}

bool parse_double_strict(const std::string& text, double& out) {
    try {
        size_t idx = 0;
        std::string s = trim_copy(text);
        if (s.empty()) return false;

        out = std::stod(s, &idx);
        while (idx < s.size() && std::isspace(static_cast<unsigned char>(s[idx]))) {
            ++idx;
        }
        return idx == s.size();
    } catch (...) {
        return false;
    }
}

bool parse_size_t_strict(const std::string& text, size_t& out) {
    try {
        size_t idx = 0;
        std::string s = trim_copy(text);
        if (s.empty()) return false;

        unsigned long long v = std::stoull(s, &idx);
        while (idx < s.size() && std::isspace(static_cast<unsigned char>(s[idx]))) {
            ++idx;
        }
        if (idx != s.size()) return false;

        out = static_cast<size_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

void set_default_newton_search_config(NewtonSearchConfig& cfg) {
    cfg.lower_bound = Vector<double>();
    cfg.upper_bound = Vector<double>();
    cfg.grid_resolution = 5;
}

void set_default_newton_numeric_config(NewtonNumericConfig& cfg) {
    cfg.max_iter = 50;
    cfg.grad_tol = 1e-6;
    cfg.step_tol = 1e-8;
    cfg.stationarity_tol = 1e-6;
    cfg.duplicate_tol = 1e-4;

    cfg.armijo_c1 = 1e-4;
    cfg.backtracking_beta = 0.5;
    cfg.min_alpha = 1e-12;

    cfg.initial_regularization = 1e-8;
    cfg.regularization_growth = 10.0;
    cfg.max_regularization = 1e8;
}

/// @brief Parse XML config from file into string.
static std::string read_file_to_string(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw ConfigFileOpenException(filename);
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

/// @brief Extract inner text of the first <tag>...</tag>.
static bool get_tag_string(const std::string& xml, const std::string& tag, std::string& out) {
    std::string open = "<" + tag + ">";
    std::string close = "</" + tag + ">";

    auto p1 = xml.find(open);
    auto p2 = xml.find(close);

    if (p1 == std::string::npos || p2 == std::string::npos || p2 <= p1) {
        return false;
    }

    p1 += open.size();
    out = xml.substr(p1, p2 - p1);
    out = trim_copy(out);
    return true;
}

static bool get_tag_double(const std::string& xml, const std::string& tag, double& out) {
    std::string s;
    if (!get_tag_string(xml, tag, s)) return false;
    return parse_double_strict(s, out);
}

static bool get_tag_size_t(const std::string& xml, const std::string& tag, size_t& out) {
    std::string s;
    if (!get_tag_string(xml, tag, s)) return false;
    return parse_size_t_strict(s, out);
}

static bool get_tag_vector_double(
    const std::string& xml,
    const std::string& tag,
    Vector<double>& out
) {
    std::string s;
    if (!get_tag_string(xml, tag, s))
        return false;
    for (char& ch : s) {
        if (ch == ',' || ch == ';')
            ch = ' ';
    }
    std::stringstream iss(s);
    std::vector<double> values;
    std::string token;
    while (iss >> token) {
        double v;
        if (!parse_double_strict(token, v))
            return false;
        values.push_back(v);
    }
    if (values.empty())
        return false;
    out = Vector<double>(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = values[i];
    }
    return true;
}

NewtonSearchConfig load_newton_search_config_from_xml(const std::string& filename) {
    NewtonSearchConfig cfg;
    set_default_newton_search_config(cfg);

    std::string xml = read_file_to_string(filename);

    if (!get_tag_vector_double(xml, "lower_bound", cfg.lower_bound)) {
        throw ConfigParseException("Failed to read lower_bound from config");
    }

    if (!get_tag_vector_double(xml, "upper_bound", cfg.upper_bound)) {
        throw ConfigParseException("Failed to read upper_bound from config");
    }

    if (!get_tag_size_t(xml, "grid_resolution", cfg.grid_resolution)) {
        throw ConfigParseException("Failed to read grid_resolution from config");
    }

    if (cfg.lower_bound.size() != cfg.upper_bound.size()) {
        throw ConfigParseException("lower_bound and upper_bound must have the same dimension");
    }

    if (cfg.lower_bound.size() == 0) {
        throw ConfigParseException("lower_bound and upper_bound must not be empty");
    }

    return cfg;
}

NewtonNumericConfig load_newton_numeric_config_from_xml(const std::string& filename) {
    NewtonNumericConfig cfg;
    set_default_newton_numeric_config(cfg);

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

    if (!get_tag_double(xml, "initial_regularization", cfg.initial_regularization)) {
        throw ConfigParseException("Failed to read initial_regularization from config");
    }

    if (!get_tag_double(xml, "regularization_growth", cfg.regularization_growth)) {
        throw ConfigParseException("Failed to read regularization_growth from config");
    }

    if (!get_tag_double(xml, "max_regularization", cfg.max_regularization)) {
        throw ConfigParseException("Failed to read max_regularization from config");
    }

    if (!get_tag_double(xml, "gradient_step", cfg.gradient_step)) {
        throw ConfigParseException("Failed to read gradient_step from config");
    }

    if (!get_tag_double(xml, "hessian_step", cfg.hessian_step)) {
        throw ConfigParseException("Failed to read hessian_step from config");
    }

    return cfg;
}
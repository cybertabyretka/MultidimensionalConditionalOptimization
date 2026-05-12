#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "exceptions/latex_parser_exceptions.hpp"

struct Monomial {
    double coeff = 0.0;
    std::unordered_map<std::string, int> powers;
};

inline bool is_ident_start(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

inline bool is_ident_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

inline std::string remove_spaces(std::string s) {
    s.erase(
        std::remove_if(s.begin(), s.end(), [](unsigned char ch) { return std::isspace(ch); }),
        s.end()
    );
    return s;
}

inline std::vector<std::string> split_terms(const std::string& expr) {
    std::vector<std::string> terms;
    std::string current;
    int brace_depth = 0;

    for (size_t i = 0; i < expr.size(); ++i) {
        char c = expr[i];
        if (c == '{') {
            ++brace_depth;
            current.push_back(c);
            continue;
        }
        if (c == '}') {
            --brace_depth;
            current.push_back(c);
            continue;
        }
        if ((c == '+' || c == '-') && brace_depth == 0) {
            if (!current.empty()) {
                terms.push_back(current);
                current.clear();
            }
            current.push_back(c);
        } else {
            current.push_back(c);
        }
    }

    if (!current.empty()) {
        terms.push_back(current);
    }

    return terms;
}

inline int parse_exponent(const std::string& s, size_t& i) {
    if (i >= s.size() || s[i] != '^') {
        return 1;
    }

    ++i;

    if (i < s.size() && s[i] == '{') {
        ++i;
        size_t start = i;
        while (i < s.size() && s[i] != '}') {
            ++i;
        }
        if (i >= s.size()) {
            throw LaTeXParserException("Invalid exponent syntax: missing '}'");
        }
        std::string exp_str = s.substr(start, i - start);
        ++i;
        try {
            return std::stoi(exp_str);
        } catch (...) {
            throw LaTeXParserException("Invalid exponent: " + exp_str);
        }
    }

    size_t start = i;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
        ++i;
    }
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        ++i;
    }

    if (start == i) {
        throw LaTeXParserException("Invalid exponent syntax");
    }

    std::string exp_str = s.substr(start, i - start);
    try {
        return std::stoi(exp_str);
    } catch (...) {
        throw LaTeXParserException("Invalid exponent: " + exp_str);
    }
}

inline Monomial parse_term(const std::string& raw_term) {
    if (raw_term.empty()) {
        throw LaTeXParserException("Empty term");
    }

    std::string s = remove_spaces(raw_term);
    size_t i = 0;

    int sign = 1;
    if (s[i] == '+' || s[i] == '-') {
        if (s[i] == '-') {
            sign = -1;
        }
        ++i;
    }

    Monomial m;
    m.coeff = 1.0;

    if (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.')) {
        size_t consumed = 0;
        try {
            m.coeff = std::stod(s.substr(i), &consumed);
        } catch (...) {
            throw LaTeXParserException("Invalid coefficient in term: " + s);
        }
        i += consumed;
    }

    while (i < s.size()) {
        if (s[i] == '*') {
            ++i;
            continue;
        }

        if (!is_ident_start(s[i])) {
            throw LaTeXParserException(std::string("Unexpected character in term: ") + s[i]);
        }

        size_t start = i;
        ++i;
        while (i < s.size() && is_ident_char(s[i])) {
            ++i;
        }
        std::string var = s.substr(start, i - start);

        int exp = parse_exponent(s, i);
        if (exp < 0) {
            throw LaTeXParserException("Negative exponents are not supported: " + var);
        }

        m.powers[var] += exp;
    }

    m.coeff *= sign;
    return m;
}

inline std::vector<Monomial> parse_polynomial(const std::string& expr) {
    std::string s = remove_spaces(expr);
    std::vector<Monomial> terms;

    for (const auto& term_str : split_terms(s)) {
        if (!term_str.empty()) {
            terms.push_back(parse_term(term_str));
        }
    }

    return terms;
}

inline std::vector<std::string> extract_variables_in_order(const std::string& expr) {
    std::string s = remove_spaces(expr);
    std::vector<std::string> variables;

    for (size_t i = 0; i < s.size();) {
        if (!is_ident_start(s[i])) {
            ++i;
            continue;
        }

        if ((s[i] == 'e' || s[i] == 'E') && i > 0) {
            const char prev = s[i - 1];
            if ((std::isdigit(static_cast<unsigned char>(prev)) || prev == '.') && i + 1 < s.size()) {
                const char next = s[i + 1];
                if (std::isdigit(static_cast<unsigned char>(next)) || next == '+' || next == '-') {
                    ++i;
                    continue;
                }
            }
        }

        size_t start = i;
        ++i;
        while (i < s.size() && is_ident_char(s[i])) {
            ++i;
        }

        std::string var = s.substr(start, i - start);
        if (std::find(variables.begin(), variables.end(), var) == variables.end()) {
            variables.push_back(var);
        }
    }

    return variables;
}

using MultiVarFunction = std::function<double(const std::vector<double>&)>;

inline MultiVarFunction parse_function(
    const std::string& latex_expr,
    const std::vector<std::string>& variables
) {
    std::vector<Monomial> terms = parse_polynomial(latex_expr);

    std::unordered_map<std::string, size_t> var_index;
    for (size_t i = 0; i < variables.size(); ++i) {
        var_index[variables[i]] = i;
    }

    return [terms = std::move(terms), var_index = std::move(var_index), variables](const std::vector<double>& values) -> double {
        if (values.size() != variables.size()) {
            throw std::invalid_argument("Wrong number of variable values");
        }

        double result = 0.0;

        for (const auto& term : terms) {
            double term_value = term.coeff;
            for (const auto& [var, exp] : term.powers) {
                auto it = var_index.find(var);
                if (it == var_index.end()) {
                    throw LaTeXParserException("Unknown variable: " + var);
                }
                term_value *= std::pow(values[it->second], exp);
            }
            result += term_value;
        }

        return result;
    };
}

#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "utils/vector.hpp"
#include "exceptions/config_exceptions.hpp"

inline std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };

    auto it1 = std::find_if(s.begin(), s.end(), not_space);
    auto it2 = std::find_if(s.rbegin(), s.rend(), not_space).base();

    if (it1 >= it2) {
        return {};
    }
    return std::string(it1, it2);
}

inline bool parse_double_strict(const std::string& text, double& out) {
    try {
        size_t idx = 0;
        std::string s = trim_copy(text);
        if (s.empty()) {
            return false;
        }

        out = std::stod(s, &idx);
        while (idx < s.size() && std::isspace(static_cast<unsigned char>(s[idx]))) {
            ++idx;
        }
        return idx == s.size();
    } catch (...) {
        return false;
    }
}

inline bool parse_size_t_strict(const std::string& text, size_t& out) {
    try {
        size_t idx = 0;
        std::string s = trim_copy(text);
        if (s.empty()) {
            return false;
        }

        unsigned long long v = std::stoull(s, &idx);
        while (idx < s.size() && std::isspace(static_cast<unsigned char>(s[idx]))) {
            ++idx;
        }
        if (idx != s.size()) {
            return false;
        }

        out = static_cast<size_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

inline std::string read_file_to_string(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw ConfigFileOpenException(filename);
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

inline bool get_tag_string(const std::string& xml, const std::string& tag, std::string& out) {
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

inline bool get_tag_double(const std::string& xml, const std::string& tag, double& out) {
    std::string s;
    if (!get_tag_string(xml, tag, s)) {
        return false;
    }
    return parse_double_strict(s, out);
}

inline bool get_tag_size_t(const std::string& xml, const std::string& tag, size_t& out) {
    std::string s;
    if (!get_tag_string(xml, tag, s)) {
        return false;
    }
    return parse_size_t_strict(s, out);
}

inline bool get_tag_vector_double(const std::string& xml, const std::string& tag, Vector<double>& out) {
    std::string s;
    if (!get_tag_string(xml, tag, s)) {
        return false;
    }

    for (char& ch : s) {
        if (ch == ',' || ch == ';') {
            ch = ' ';
        }
    }

    std::stringstream iss(s);
    std::vector<double> values;
    std::string token;
    while (iss >> token) {
        double v = 0.0;
        if (!parse_double_strict(token, v)) {
            return false;
        }
        values.push_back(v);
    }

    if (values.empty()) {
        return false;
    }

    out = Vector<double>(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = values[i];
    }
    return true;
}

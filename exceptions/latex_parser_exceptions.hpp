#pragma once

#include <stdexcept>
#include <string>

/// @brief Exception class for errors encountered during LaTeX parsing.
class LaTeXParserException : public std::runtime_error {
public:
    explicit LaTeXParserException(const std::string& message)
        : std::runtime_error(message) {}
};
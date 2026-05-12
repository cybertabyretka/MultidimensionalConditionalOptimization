#pragma once

#include <stdexcept>

class NumericalError : public std::runtime_error {
public:
    explicit NumericalError(const std::string& detail)
        : std::runtime_error("Numerical Error: " + detail) {}
};
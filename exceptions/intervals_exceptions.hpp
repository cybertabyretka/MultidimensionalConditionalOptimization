#pragma once

#include <stdexcept>

class InvalidIntervalError : public std::runtime_error {
public:
    explicit InvalidIntervalError(const std::string& detail)
        : std::runtime_error("Invalid Interval: " + detail) {}
};
#pragma once

#include <stdexcept>
#include <string>

class ProjectedGradientConfigError : public std::runtime_error {
public:
    explicit ProjectedGradientConfigError(const std::string& message)
        : std::runtime_error(message) {}
};

class ProjectedGradientOptimizationError : public std::runtime_error {
public:
    explicit ProjectedGradientOptimizationError(const std::string& message)
        : std::runtime_error(message) {}
};

class ProjectedGradientConvergenceError : public std::runtime_error {
public:
    explicit ProjectedGradientConvergenceError(const std::string& message)
        : std::runtime_error(message) {}
};

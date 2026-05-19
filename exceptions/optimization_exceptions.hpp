#pragma once

#include <stdexcept>
#include <string>

class OptimizationError : public std::runtime_error {
public:
    explicit OptimizationError(const std::string& message) 
        : std::runtime_error("Optimization Error: " + message) {}
};

class InputOptimizationError : public OptimizationError {
public:
    explicit InputOptimizationError(const std::string& message)
        : OptimizationError("Input Optimization Error: " + message) {}
};

class ConvergenceError : public OptimizationError {
public:
    explicit ConvergenceError(const std::string& message)
        : OptimizationError("Convergence Error: " + message) {}
};

class ObjectiveEvaluationError : public OptimizationError {
public:
    explicit ObjectiveEvaluationError(const std::string& message)
        : OptimizationError("Objective Evaluation Optimiazation Error: " + message) {}
};

class ProjectionError : public OptimizationError {
public:
    explicit ProjectionError(const std::string& message)
        : OptimizationError("Projection Error: " + message) {}
};

class ProjectedGradientConfigError : public ProjectionError {
public:
    explicit ProjectedGradientConfigError(const std::string& message)
        : ProjectionError("Projected Gradient Config Error: " + message) {}
};

class ProjectedGradientOptimizationError : public ProjectionError {
public:
    explicit ProjectedGradientOptimizationError(const std::string& message)
        : ProjectionError("Projected Gradient Optimization Error: " + message) {}
};

class ProjectedGradientConvergenceError : public ProjectionError {
public:
    explicit ProjectedGradientConvergenceError(const std::string& message)
        : ProjectionError("Projected Gradient Convergence Error: " + message) {}
};

#pragma once

#include <stdexcept>

class OptimizationError : public std::runtime_error {
public:
    explicit OptimizationError(const std::string& message) 
        : std::runtime_error("Optimization Error: " + message) {}
};

class InputOptimizationError : public OptimizationError {
public:
    explicit InputOptimizationError(const std::string& detail)
        : OptimizationError("Input Optimization Error: " + detail) {}
};

class ConvergenceError : public OptimizationError {
public:
    explicit ConvergenceError(const std::string& detail)
        : OptimizationError("Convergence Error: " + detail) {}
};

class ObjectiveEvaluationError : public OptimizationError {
public:
    explicit ObjectiveEvaluationError(const std::string& detail)
        : OptimizationError("Objective Evaluation Error: " + detail) {}
};
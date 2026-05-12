#pragma once

#include <stdexcept>

class DimensionMismatchError : public std::runtime_error {
public:
    explicit DimensionMismatchError(const std::string& detail)
        : std::runtime_error("Dimension Mismatch: " + detail) {}
};

class SingularMatrixError : public std::runtime_error {
public:
    explicit SingularMatrixError(const std::string& detail)
        : std::runtime_error("Singular Matrix: " + detail) {}
};
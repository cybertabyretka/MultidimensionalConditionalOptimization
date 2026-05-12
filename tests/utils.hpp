#pragma once

#include <cmath>
#include <iostream>

void print_test_failed(const std::string& test_name, const std::string& reason) {
    std::cerr << "FAIL: " << test_name << " - " << reason << '\n';
}

bool double_equals(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}
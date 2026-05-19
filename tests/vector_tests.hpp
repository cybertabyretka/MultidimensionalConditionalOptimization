#pragma once

#include "utils/vector.hpp"

#include "tests/utils.hpp"

#include "exceptions/vector_matrix_exceptions.hpp"
#include "exceptions/numerical_exceptions.hpp"


/**
 * @brief Vector - Normal case
 * 
 * Tests basic functionality: construction, operations, dot product, norm
 */
void test_vector_normal() {
    try {
        Vector<double> v1{1.0, 2.0, 3.0};
        Vector<double> v2{4.0, 5.0, 6.0};
        // Check size
        if (v1.size() != 3) throw std::logic_error("Size mismatch");
        // Check element access
        if (v1[0] != 1.0 || v1[1] != 2.0 || v1[2] != 3.0) 
            throw std::logic_error("Element access failed");
        // Addition: [1,2,3] + [4,5,6] = [5,7,9]
        Vector<double> sum = v1 + v2;
        if (!sum.equals(Vector<double>{5.0, 7.0, 9.0}, 1e-9))
            throw std::logic_error("Addition failed");
        // Subtraction: [4,5,6] - [1,2,3] = [3,3,3]
        Vector<double> diff = v2 - v1;
        if (!diff.equals(Vector<double>{3.0, 3.0, 3.0}, 1e-9))
            throw std::logic_error("Subtraction failed");
        // Unary minus: -[1,2,3] = [-1,-2,-3]
        Vector<double> neg = -v1;
        if (!neg.equals(Vector<double>{-1.0, -2.0, -3.0}, 1e-9))
            throw std::logic_error("Negation failed");
        // Scalar multiplication: [1,2,3] * 2 = [2,4,6]
        Vector<double> scaled = v1 * 2.0;
        if (!scaled.equals(Vector<double>{2.0, 4.0, 6.0}, 1e-9))
            throw std::logic_error("Scalar multiplication failed");
        // Dot product: [1,2,3] · [4,5,6] = 32
        double dot_product = v1.dot(v2);
        if (!double_equals(dot_product, 32.0))
            throw std::logic_error("Dot product failed");
        // Norm: ||[1,2,3]|| = sqrt(14)
        double norm = v1.norm();
        double expected_norm = std::sqrt(14.0);
        if (!double_equals(norm, expected_norm))
            throw std::logic_error("Norm calculation failed");
        // Division: [4,2,6] / 2 = [2,1,3]
        Vector<double> divided = Vector<double>{4.0, 2.0, 6.0} / 2.0;
        if (!divided.equals(Vector<double>{2.0, 1.0, 3.0}, 1e-9))
            throw std::logic_error("Division failed");
    } catch (const std::logic_error& e) {
        print_test_failed("Vector_Normal", e.what());
    }
}

/**
 * @brief Vector - Edge and abnormal cases
 * 
 * Tests error handling: empty vector, out-of-bounds access, incompatible sizes
 */
void test_vector_edge_cases() {
    try {
        // Empty vector
        Vector<double> empty;
        if (empty.size() != 0) throw std::logic_error("Empty vector size should be 0");
        if (!empty.empty()) throw std::logic_error("Empty vector should report empty()");
        // Out-of-bounds access exception
        Vector<double> v{1.0, 2.0, 3.0};
        try {
            v.at(5);  // Should throw out-of-bounds access exception
            throw std::logic_error("Should have thrown out_of_range");
        } catch (const std::out_of_range&) {}
        // Incompatible sizes for addition
        Vector<double> v1{1.0, 2.0};
        Vector<double> v2{1.0, 2.0, 3.0};
        try {
            v1 + v2;  // Should throw DimensionMismatchError
            throw std::logic_error("Should have thrown DimensionMismatchError for addition");
        } catch (const DimensionMismatchError&) {}
        // Division by zero
        Vector<double> v3{1.0, 2.0, 3.0};
        try {
            v3 / 0.0;  // Should throw NumericalError
            throw std::logic_error("Should have thrown NumericalError for division by zero");
        } catch (const NumericalError&) {}
        // Dot product with incompatible sizes
        try {
            v1.dot(v2);  // Should throw DimensionMismatchError
            throw std::logic_error("Should have thrown DimensionMismatchError for dot product");
        } catch (const DimensionMismatchError&) {}
    } catch (const std::logic_error& e) {
        print_test_failed("Vector_EdgeCases", e.what());
    }
}
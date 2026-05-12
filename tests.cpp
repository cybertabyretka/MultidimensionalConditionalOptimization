#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>
#include <numbers>

#include "newton_optimizer.hpp"
#include "newton_configs.hpp"

#include "utils/vector.hpp"
#include "utils/matrix.hpp"

#include "tests/vector_tests.hpp"
#include "tests/matrix_tests.hpp"
#include "tests/optimizer_tests.hpp"

#include "exceptions/vector_matrix_exceptions.hpp"
#include "exceptions/numerical_exceptions.hpp"
#include "exceptions/optimization_exceptions.hpp"

int main() {
    // Vector tests
    test_vector_normal();
    test_vector_edge_cases();
    
    // Matrix tests
    test_matrix_normal();
    test_matrix_edge_cases();
    
    // Optimizer tests
    test_optimizer_quadratic();
    test_optimizer_rosenbrock();
    test_optimizer_saddle_point();
    test_optimizer_multiple_minima();
    
    // Derivatives tests
    test_numerical_gradient();
    test_numerical_hessian();
    return 0;
}

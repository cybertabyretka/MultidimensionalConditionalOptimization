#include <vector>

#include "utils/grid_generator.hpp"
#include "utils/vector.hpp"

#include "exceptions/vector_matrix_exceptions.hpp"
#include "exceptions/intervals_exceptions.hpp"

std::vector<Vector<double>> GridGenerator::generate(
    const Vector<double>& lower,
    const Vector<double>& upper,
    size_t resolution
) const {
    if (lower.size() != upper.size()) {
        throw DimensionMismatchError("Bounds dimension mismatch");
    }
    if (lower.empty()) {
        throw DimensionMismatchError("Bounds vector is empty");
    }
    if (resolution < 2) {
        throw InvalidIntervalError("Grid resolution must be at least 2");
    }

    const size_t n = lower.size();
    for (size_t i = 0; i < n; ++i) {
        if (lower[i] >= upper[i]) {
            throw InvalidIntervalError("Lower bound must be strictly less than upper bound");
        }
    }

    std::vector<Vector<double>> points;
    std::vector<size_t> index(n, 0);
    std::vector<double> step(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        step[i] = (upper[i] - lower[i]) / static_cast<double>(resolution - 1);
    }

    while (true) {
        Vector<double> pt(n);
        for (size_t d = 0; d < n; ++d) {
            pt[d] = lower[d] + static_cast<double>(index[d]) * step[d];
        }
        points.push_back(pt);

        size_t pos = n;
        while (pos > 0) {
            --pos;
            ++index[pos];
            if (index[pos] < resolution) break;
            index[pos] = 0;
        }
        if (pos == 0 && index[0] == 0) break;
    }
    return points;
}

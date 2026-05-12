#pragma once

#include <vector>

#include "utils/vector.hpp"

/**
 * @brief Generates a regular grid of points inside a hyper-rectangle.
 *
 * The grid is defined by lower and upper bounds for each dimension.
 * Each dimension is sampled with the same number of points: resolution.
 */
class GridGenerator {
public:
    /**
     * @brief Build all grid points for the given bounds.
     *
     * @param lower Lower bounds for each dimension
     * @param upper Upper bounds for each dimension
     * @param resolution Number of points per dimension
     * @return Flat list of all grid points
     * @throw DimensionMismatchError
     * @throw InvalidIntervalError
     */
    std::vector<Vector<double>> generate(
        const Vector<double>& lower,
        const Vector<double>& upper,
        size_t resolution
    ) const;
};
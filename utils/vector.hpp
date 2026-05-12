#pragma once

#include <cmath>
#include <initializer_list>
#include <numeric>
#include <string>
#include <ostream>
#include <type_traits>
#include <vector>

#include "exceptions/vector_matrix_exceptions.hpp"
#include "exceptions/numerical_exceptions.hpp"

/**
 * @brief A lightweight numeric vector with basic linear algebra operations.
 * @tparam T Arithmetic element type, excluding bool.
 */
template <typename T>
class Vector {
    static_assert(
        std::is_arithmetic_v<std::remove_cv_t<T>> &&
        !std::is_same_v<std::remove_cv_t<T>, bool>,
        "Vector<T>: T must be a real or integer numeric type, excluding complex/bool"
    );

    std::vector<T> data_;

    /**
     * @brief Checks that an index is within vector bounds.
     * @param idx Index to validate.
     * @throw std::out_of_range If idx is outside the vector range.
     */
    void check_index(size_t idx) const {
        if (idx >= data_.size()) {
            throw std::out_of_range("Vector index out of range: " + std::to_string(idx));
        }
    }

    /**
     * @brief Checks that two vectors have equal dimension.
     * @param other Vector to compare against.
     * @param op Human-readable operation name used in the error message.
     * @throw DimensionMismatchError If the vector sizes differ.
     */
    void check_dimension(const Vector& other, const std::string& op) const {
        if (data_.size() != other.size()) {
            throw DimensionMismatchError("Vectors must have same size for " + op);
        }
    }

public:
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    /**
     * @brief Constructs a vector of given size filled with a value.
     * @param n Number of elements.
     * @param val Initial value for each element.
     */
    explicit Vector(size_t n = 0, T val = T{}) : data_(n, val) {}

    /**
     * @brief Constructs a vector from an initializer list.
     * @param list Element values.
     */
    explicit Vector(std::initializer_list<T> list) : data_(list) {}

    /**
     * @brief Returns the number of elements.
     * @return Vector size.
     */
    size_t size() const noexcept { return data_.size(); }

    /**
     * @brief Checks whether the vector contains no elements.
     * @return true if the vector is empty, false otherwise.
     */
    bool empty() const noexcept { return data_.empty(); }

    /**
     * @brief Returns a mutable iterator to the first element.
     * @return Iterator to beginning.
     */
    iterator begin() noexcept { return data_.begin(); }

    /**
     * @brief Returns a mutable iterator to the past-the-end element.
     * @return Iterator to end.
     */
    iterator end() noexcept { return data_.end(); }

    /**
     * @brief Returns a constant iterator to the first element.
     * @return Const iterator to beginning.
     */
    const_iterator begin() const noexcept { return data_.begin(); }

    /**
     * @brief Returns a constant iterator to the past-the-end element.
     * @return Const iterator to end.
     */
    const_iterator end() const noexcept { return data_.end(); }

    /**
     * @brief Returns a constant iterator to the first element.
     * @return Const iterator to beginning.
     */
    const_iterator cbegin() const noexcept { return data_.cbegin(); }

    /**
     * @brief Returns a constant iterator to the past-the-end element.
     * @return Const iterator to end.
     */
    const_iterator cend() const noexcept { return data_.cend(); }

    /**
     * @brief Accesses an element with bounds checking.
     * @param idx Element index.
     * @return Constant reference to the element.
     * @throw std::out_of_range If idx is outside the vector range.
     */
    const T& at(size_t idx) const { check_index(idx); return data_[idx]; }
    /**
     * @brief Accesses an element with bounds checking.
     * @param idx Element index.
     * @return Mutable reference to the element.
     * @throw std::out_of_range If idx is outside the vector range.
     */
    T& at(size_t idx) { check_index(idx); return data_[idx]; }

    const T& operator[](size_t idx) const { return data_[idx]; }
    T& operator[](size_t idx) { return data_[idx]; }

    bool operator==(const Vector& other) const noexcept { return data_ == other.data_; }

    bool operator!=(const Vector& other) const noexcept { return !(*this == other); }

    /**
     * @brief Compares vectors using a tolerance.
     * @param other Vector to compare against.
     * @param tol Absolute tolerance used for element-wise comparison.
     * @return true if all elements are within tolerance and sizes match.
     */
    bool equals(const Vector& other, double tol = 1e-9) const {
        if (size() != other.size()) return false;
        for (size_t i = 0; i < size(); ++i) {
            if (std::abs(data_[i] - other[i]) > static_cast<T>(tol)) {
                return false;
            }
        }
        return true;
    }

    Vector operator+(const Vector& other) const {
        check_dimension(other, "addition");
        Vector res(size());
        for (size_t i = 0; i < size(); ++i) res[i] = data_[i] + other[i];
        return res;
    }

    Vector operator-(const Vector& other) const {
        check_dimension(other, "subtraction");
        Vector res(size());
        for (size_t i = 0; i < size(); ++i) res[i] = data_[i] - other[i];
        return res;
    }

    Vector operator-() const {
        Vector res(size());
        for (size_t i = 0; i < size(); ++i) res[i] = -data_[i];
        return res;
    }

    Vector operator*(T scalar) const {
        Vector res(size());
        for (size_t i = 0; i < size(); ++i) res[i] = data_[i] * scalar;
        return res;
    }

    Vector operator/(T scalar) const {
        if (scalar == T{}) {
            throw NumericalError("Division by zero in vector");
        }
        Vector res(size());
        for (size_t i = 0; i < size(); ++i) res[i] = data_[i] / scalar;
        return res;
    }

    friend Vector operator*(T scalar, const Vector& v) {
        return v * scalar;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
        os << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i != 0) os << ", ";
            os << v[i];
        }
        os << "]";
        return os;
    }

    /**
     * @brief Computes the dot product with another vector.
     * @param other Right-hand vector.
     * @return Dot product value.
     * @throw DimensionMismatchError If vector sizes differ.
     */
    T dot(const Vector& other) const {
        check_dimension(other, "dot product");
        return std::inner_product(data_.begin(), data_.end(), other.data_.begin(), T{});
    }

    /**
     * @brief Computes the Euclidean norm.
     * @return Vector magnitude.
     */
    auto norm() const {
        using ResultType = std::common_type_t<T, double>;
        return std::sqrt(static_cast<ResultType>(dot(*this)));
    }
};
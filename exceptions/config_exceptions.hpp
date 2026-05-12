#pragma once

#include <stdexcept>
#include <string>

/// @brief Base exception class for configuration-related errors.
class ConfigException : public std::runtime_error {
public:
    explicit ConfigException(const std::string& message)
        : std::runtime_error(message) {}
};

/// @brief Exception thrown when a configuration file cannot be opened.
class ConfigFileOpenException : public ConfigException {
public:
    explicit ConfigFileOpenException(const std::string& filename)
        : ConfigException("Could not open config file: " + filename) {}
};

/// @brief Exception thrown when a configuration variable cannot be parsed or is missing.
class ConfigParseException : public ConfigException {
public:
    explicit ConfigParseException(const std::string& message)
        : ConfigException(message) {}
};
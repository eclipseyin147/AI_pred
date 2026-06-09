#ifndef DATA_READER_H
#define DATA_READER_H

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

inline std::string trim_copy(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

inline bool parse_double_token(const std::string& token, double& out) {
    std::string trimmed = trim_copy(token);
    if (trimmed.empty()) {
        return false;
    }
    char* end_ptr = nullptr;
    const char* begin = trimmed.c_str();
    out = std::strtod(begin, &end_ptr);
    if (begin == end_ptr) {
        return false;
    }
    while (*end_ptr != '\0') {
        if (!std::isspace(static_cast<unsigned char>(*end_ptr))) {
            return false;
        }
        ++end_ptr;
    }
    return true;
}

inline bool parse_numeric_row(const std::string& line, bool is_csv, std::vector<double>& row) {
    row.clear();

    if (is_csv) {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            double value = 0.0;
            if (!parse_double_token(token, value)) {
                row.clear();
                return false;
            }
            row.push_back(value);
        }
        return !row.empty();
    }

    std::stringstream ss(line);
    double value = 0.0;
    while (ss >> value) {
        row.push_back(value);
    }
    if (row.empty()) {
        return false;
    }
    ss >> std::ws;
    return ss.eof();
}

// Read all numeric data rows (skip first non-numeric line as header; ignore blank lines).
inline std::vector<std::vector<double>> readAllNumericRows(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string raw_line;
    bool detected_format = false;
    bool is_csv = false;
    bool first_non_empty_line = true;

    while (std::getline(file, raw_line)) {
        if (first_non_empty_line && raw_line.size() >= 3 &&
            static_cast<unsigned char>(raw_line[0]) == 0xEF &&
            static_cast<unsigned char>(raw_line[1]) == 0xBB &&
            static_cast<unsigned char>(raw_line[2]) == 0xBF) {
            raw_line.erase(0, 3);
        }

        const std::string line = trim_copy(raw_line);
        if (line.empty()) {
            continue;
        }

        if (!detected_format) {
            is_csv = (line.find(',') != std::string::npos);
            detected_format = true;
        }

        std::vector<double> row;
        if (!parse_numeric_row(line, is_csv, row)) {
            if (first_non_empty_line) {
                first_non_empty_line = false;
                continue;
            }
            continue;
        }

        first_non_empty_line = false;
        data.push_back(row);
    }

    return data;
}

// numRowsBegin/numRowsEnd: 1-based inclusive indices of numeric data rows (GUI row numbers).
// numRowsBegin <= 0 is treated as 1. numRowsEnd <= 0 reads through the last data row.
inline std::vector<std::vector<double>> readDataFile(const std::string& filename,
                                                     int numRowsBegin,
                                                     int numRowsEnd) {
    const auto all_rows = readAllNumericRows(filename);
    if (all_rows.empty()) {
        return all_rows;
    }

    const int start_row = (numRowsBegin <= 0) ? 1 : numRowsBegin;
    const int end_row = (numRowsEnd <= 0)
                            ? static_cast<int>(all_rows.size())
                            : numRowsEnd;

    if (start_row < 1 || start_row > end_row) {
        return {};
    }

    const size_t from = static_cast<size_t>(start_row - 1);
    if (from >= all_rows.size()) {
        return {};
    }

    const size_t to = std::min(static_cast<size_t>(end_row), all_rows.size());
    return std::vector<std::vector<double>>(all_rows.begin() + static_cast<std::ptrdiff_t>(from),
                                            all_rows.begin() + static_cast<std::ptrdiff_t>(to));
}

// Read first numRows numeric data rows (1-based count from first data row after header).
inline std::vector<std::vector<double>> readDataFile(const std::string& filename, int numRows) {
    if (numRows <= 0) {
        return readAllNumericRows(filename);
    }
    return readDataFile(filename, 1, numRows);
}

#endif // DATA_READER_H

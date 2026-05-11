#ifndef COMMON_FFN_H
#define COMMON_FFN_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <memory>

// ============================================================================
// Generalized Feedforward Neural Network
// ============================================================================
struct FeedForwardNet : torch::nn::Module {
    std::vector<torch::nn::Linear> hidden_layers;
    torch::nn::Linear output_layer{nullptr};

    FeedForwardNet(int64_t input_size, const std::vector<int64_t>& hidden_sizes, int64_t output_size = 1) {
        int64_t prev = input_size;
        for (size_t i = 0; i < hidden_sizes.size(); ++i) {
            auto layer = register_module("fc" + std::to_string(i + 1), torch::nn::Linear(prev, hidden_sizes[i]));
            hidden_layers.push_back(layer);
            prev = hidden_sizes[i];
        }
        output_layer = register_module("fc" + std::to_string(hidden_sizes.size() + 1), torch::nn::Linear(prev, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        for (auto& layer : hidden_layers) {
            x = torch::sigmoid(layer->forward(x));
        }
        x = output_layer->forward(x);
        return x;
    }
};

// ============================================================================
// Normalization / Standardization Strategy
// ============================================================================
enum class NormalizationMethod {
    NONE,
    MINMAX_NEG1_1,   // MATLAB mapminmax style: [-1, 1]
    MINMAX_0_1,      // Classic Min-Max: [0, 1]
    Z_SCORE          // StandardScaler: (x - mean) / std
};

inline NormalizationMethod parse_normalization_method(const std::string& name) {
    if (name == "none" || name == "disabled") return NormalizationMethod::NONE;
    if (name == "minmax_0_1") return NormalizationMethod::MINMAX_0_1;
    if (name == "z_score") return NormalizationMethod::Z_SCORE;
    return NormalizationMethod::MINMAX_NEG1_1;  // default
}

// Abstract base
class DataNormalizer {
public:
    virtual ~DataNormalizer() = default;
    virtual void fit(const torch::Tensor& X, const torch::Tensor& Y) = 0;
    virtual torch::Tensor transform_X(const torch::Tensor& X) = 0;
    virtual torch::Tensor transform_Y(const torch::Tensor& Y) = 0;
    virtual torch::Tensor inverse_transform_Y(const torch::Tensor& Y_norm) = 0;
    virtual std::string name() const = 0;
};

// Min-Max to [-1, 1] (original MATLAB mapminmax behavior)
class MinMaxScalerNeg1To1 : public DataNormalizer {
public:
    torch::Tensor x_min, x_max, y_min, y_max;

    void fit(const torch::Tensor& X, const torch::Tensor& Y) override {
        x_min = std::get<0>(X.min(1));
        x_max = std::get<0>(X.max(1));
        y_min = Y.min();
        y_max = Y.max();
    }

    torch::Tensor transform_X(const torch::Tensor& X) override {
        auto x_range = x_max - x_min;
        x_range = torch::where(x_range == 0, torch::ones_like(x_range), x_range);
        if (X.dim() == 2) {
            return 2.0 * (X - x_min.unsqueeze(1)) / x_range.unsqueeze(1) - 1.0;
        } else {
            return 2.0 * (X - x_min) / x_range - 1.0;
        }
    }

    torch::Tensor transform_Y(const torch::Tensor& Y) override {
        auto y_range = y_max - y_min;
        if (y_range.item<double>() == 0) y_range = torch::ones_like(y_range);
        return 2.0 * (Y - y_min) / y_range - 1.0;
    }

    torch::Tensor inverse_transform_Y(const torch::Tensor& Y_norm) override {
        auto y_range = y_max - y_min;
        if (y_range.item<double>() == 0) y_range = torch::ones_like(y_range);
        return (Y_norm + 1.0) * y_range / 2.0 + y_min;
    }

    std::string name() const override { return "minmax_neg1_1"; }
};

// Min-Max to [0, 1]
class MinMaxScaler0To1 : public DataNormalizer {
public:
    torch::Tensor x_min, x_max, y_min, y_max;

    void fit(const torch::Tensor& X, const torch::Tensor& Y) override {
        x_min = std::get<0>(X.min(1));
        x_max = std::get<0>(X.max(1));
        y_min = Y.min();
        y_max = Y.max();
    }

    torch::Tensor transform_X(const torch::Tensor& X) override {
        auto x_range = x_max - x_min;
        x_range = torch::where(x_range == 0, torch::ones_like(x_range), x_range);
        if (X.dim() == 2) {
            return (X - x_min.unsqueeze(1)) / x_range.unsqueeze(1);
        } else {
            return (X - x_min) / x_range;
        }
    }

    torch::Tensor transform_Y(const torch::Tensor& Y) override {
        auto y_range = y_max - y_min;
        if (y_range.item<double>() == 0) y_range = torch::ones_like(y_range);
        return (Y - y_min) / y_range;
    }

    torch::Tensor inverse_transform_Y(const torch::Tensor& Y_norm) override {
        auto y_range = y_max - y_min;
        if (y_range.item<double>() == 0) y_range = torch::ones_like(y_range);
        return Y_norm * y_range + y_min;
    }

    std::string name() const override { return "minmax_0_1"; }
};

// Z-score (StandardScaler)
class StandardScaler : public DataNormalizer {
public:
    torch::Tensor x_mean, x_std, y_mean, y_std;

    void fit(const torch::Tensor& X, const torch::Tensor& Y) override {
        x_mean = X.mean(1);
        x_std = X.std(1, true);
        y_mean = Y.mean();
        y_std = Y.std();
        x_std = torch::where(x_std == 0, torch::ones_like(x_std), x_std);
        if (y_std.item<double>() == 0) y_std = torch::ones_like(y_std);
    }

    torch::Tensor transform_X(const torch::Tensor& X) override {
        if (X.dim() == 2) {
            return (X - x_mean.unsqueeze(1)) / x_std.unsqueeze(1);
        } else {
            return (X - x_mean) / x_std;
        }
    }

    torch::Tensor transform_Y(const torch::Tensor& Y) override {
        return (Y - y_mean) / y_std;
    }

    torch::Tensor inverse_transform_Y(const torch::Tensor& Y_norm) override {
        return Y_norm * y_std + y_mean;
    }

    std::string name() const override { return "z_score"; }
};

// No-op normalizer (disabled)
class NoOpNormalizer : public DataNormalizer {
public:
    void fit(const torch::Tensor&, const torch::Tensor&) override {}
    torch::Tensor transform_X(const torch::Tensor& X) override { return X; }
    torch::Tensor transform_Y(const torch::Tensor& Y) override { return Y; }
    torch::Tensor inverse_transform_Y(const torch::Tensor& Y_norm) override { return Y_norm; }
    std::string name() const override { return "none"; }
};

// Factory
inline std::unique_ptr<DataNormalizer> create_normalizer(NormalizationMethod method) {
    switch (method) {
        case NormalizationMethod::NONE: return std::make_unique<NoOpNormalizer>();
        case NormalizationMethod::MINMAX_0_1: return std::make_unique<MinMaxScaler0To1>();
        case NormalizationMethod::Z_SCORE: return std::make_unique<StandardScaler>();
        default: return std::make_unique<MinMaxScalerNeg1To1>();
    }
}

// ============================================================================
// Read whitespace-delimited data file
// ============================================================================
inline std::vector<std::vector<double>> readDataFile(const std::string& filename, int numRows) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            data.push_back(row);
        }
        if (numRows > 0 && data.size() >= static_cast<size_t>(numRows)) {
            break;
        }
    }
    file.close();
    return data;
}

// ============================================================================
// Metrics (tensor versions)
// ============================================================================
inline double calculateRSquared(const torch::Tensor& y_true, const torch::Tensor& y_pred) {
    auto y_mean = y_true.mean();
    auto ss_tot = ((y_true - y_mean) * (y_true - y_mean)).sum();
    auto ss_res = ((y_true - y_pred) * (y_true - y_pred)).sum();
    return (1.0 - ss_res.item<double>() / ss_tot.item<double>());
}

inline double calculateRMSE(const torch::Tensor& y_true, const torch::Tensor& y_pred) {
    auto mse = ((y_true - y_pred) * (y_true - y_pred)).mean();
    return std::sqrt(mse.item<double>());
}

inline double calculateMAE(const torch::Tensor& y_true, const torch::Tensor& y_pred) {
    auto mae = torch::abs(y_true - y_pred).mean();
    return mae.item<double>();
}

// ============================================================================
// Metrics (vector versions)
// ============================================================================
inline double calculateRSquared(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double y_mean = 0.0;
    for (double val : y_true) y_mean += val;
    y_mean /= y_true.size();
    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_tot += std::pow(y_true[i] - y_mean, 2);
        ss_res += std::pow(y_true[i] - y_pred[i], 2);
    }
    return 1.0 - ss_res / ss_tot;
}

inline double calculateRMSE(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        sum += std::pow(y_true[i] - y_pred[i], 2);
    }
    return std::sqrt(sum / y_true.size());
}

inline double calculateMAE(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        sum += std::abs(y_true[i] - y_pred[i]);
    }
    return sum / y_true.size();
}

inline double calculateMeanRE(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        sum += std::abs((y_pred[i] - y_true[i]) / y_true[i]) * 100.0;
    }
    return sum / y_true.size();
}

#endif // COMMON_FFN_H

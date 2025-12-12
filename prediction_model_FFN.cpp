#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Feedforward Neural Network (similar to MATLAB's newff)
struct FeedForwardNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    FeedForwardNet(int64_t input_size, int64_t hidden1_size, int64_t hidden2_size, int64_t output_size) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden1_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden1_size, hidden2_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden2_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::sigmoid(fc1->forward(x));
        x = torch::sigmoid(fc2->forward(x));
        x = fc3->forward(x);  // Linear output for regression
        return x;
    }
};

// Min-Max normalization class (MATLAB's mapminmax normalizes to [-1, 1])
class MinMaxScaler {
public:
    torch::Tensor x_min, x_max, y_min, y_max;

    void fit(const torch::Tensor& X, const torch::Tensor& Y) {
        // X shape: [features, samples] - same as MATLAB
        x_min = std::get<0>(X.min(1));  // min across samples (dim=1)
        x_max = std::get<0>(X.max(1));  // max across samples (dim=1)
        y_min = Y.min();
        y_max = Y.max();
    }

    torch::Tensor transform_X(const torch::Tensor& X) {
        // X shape: [features, samples] or [features]
        auto x_range = x_max - x_min;
        x_range = torch::where(x_range == 0, torch::ones_like(x_range), x_range);

        if (X.dim() == 2) {
            return 2.0 * (X - x_min.unsqueeze(1)) / x_range.unsqueeze(1) - 1.0;
        } else {
            return 2.0 * (X - x_min) / x_range - 1.0;
        }
    }

    torch::Tensor transform_Y(const torch::Tensor& Y) {
        auto y_range = y_max - y_min;
        if (y_range.item<double>() == 0) y_range = torch::ones_like(y_range);
        return 2.0 * (Y - y_min) / y_range - 1.0;
    }

    torch::Tensor inverse_transform_Y(const torch::Tensor& Y_norm) {
        auto y_range = y_max - y_min;
        if (y_range.item<double>() == 0) y_range = torch::ones_like(y_range);
        return (Y_norm + 1.0) * y_range / 2.0 + y_min;
    }
};

// Read data file
std::vector<std::vector<double>> readDataFile(const std::string& filename, int numRows) {
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

        if (numRows > 0 && data.size() >= numRows) {
            break;
        }
    }

    file.close();
    return data;
}

// Calculate R-squared
double calculateRSquared(const torch::Tensor& y_true, const torch::Tensor& y_pred) {
    auto y_mean = y_true.mean();
    auto ss_tot = ((y_true - y_mean) * (y_true - y_mean)).sum();
    auto ss_res = ((y_true - y_pred) * (y_true - y_pred)).sum();
    return (1.0 - ss_res.item<double>() / ss_tot.item<double>());
}

// Calculate RMSE
double calculateRMSE(const torch::Tensor& y_true, const torch::Tensor& y_pred) {
    auto mse = ((y_true - y_pred) * (y_true - y_pred)).mean();
    return std::sqrt(mse.item<double>());
}

// Load configuration from JSON file
json loadConfig(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file '" << config_file << "'. Using defaults." << std::endl;
        // Return default configuration
        return json{
            {"optimizer", {
                {"type", "lbfgs"},
                {"lbfgs", {
                    {"learning_rate", 1.0},
                    {"max_iter", 20},
                    {"max_eval", 25},
                    {"tolerance_grad", 1e-7},
                    {"tolerance_change", 1e-9},
                    {"history_size", 100}
                }}
            }},
            {"training", {
                {"epochs", 1000},
                {"goal_loss", 1e-10},
                {"max_iterations", 10},
                {"target_r2", 0.8},
                {"print_interval", 100}
            }},
            {"model", {
                {"hidden_layer1", 50},
                {"hidden_layer2", 50}
            }},
            {"data", {
                {"window_size", 5},
                {"train_samples", 300},
                {"data_file", "Data_V13_40kW.txt"},
                {"num_rows", 900}
            }}
        };
    }

    json config;
    file >> config;
    file.close();
    return config;
}

int main(int argc, char* argv[]) {
    std::cout << std::fixed << std::setprecision(6);

    // Load configuration
    std::string config_file = "config.json";
    if (argc > 1) {
        config_file = argv[1];
    }

    json config = loadConfig(config_file);
    std::cout << "Loaded configuration from: " << config_file << std::endl;
    std::cout << "Optimizer: " << config["optimizer"]["type"] << std::endl;

    // Set OpenMP threads
    int num_threads = omp_get_max_threads();

    // Allow user to override via environment variable
    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    if (omp_env != nullptr) {
        num_threads = std::atoi(omp_env);
    }

    omp_set_num_threads(num_threads);
    std::cout << "OpenMP enabled with " << num_threads << " threads." << std::endl;

    // Configure PyTorch to use OpenMP for intra-op parallelism
    torch::set_num_threads(num_threads);
    torch::set_num_interop_threads(num_threads);
    std::cout << "PyTorch configured to use " << num_threads << " threads." << std::endl;

    // Check for CUDA availability and set device
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        std::cout << "Number of CUDA devices: " << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "CUDA not available. Training on CPU with OpenMP acceleration." << std::endl;
    }

    // Load data
    std::string data_file = config["data"]["data_file"];
    int num_rows = config["data"]["num_rows"];
    std::cout << "\nLoading data from " << data_file << "..." << std::endl;
    auto raw_data = readDataFile(data_file, num_rows);

    if (raw_data.empty()) {
        std::cerr << "Error: No data loaded!" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << raw_data.size() << " rows of data." << std::endl;

    const int w = config["data"]["window_size"];  // Window size
    const size_t dd = raw_data.size();

    // Extract Input and Output columns (MATLAB: columns 5, 6, 9, 11 are 1-indexed)
    // In C++ (0-indexed): columns 4, 5, 8, 10
    // Output: column 12 in MATLAB (1-indexed) = column 11 in C++ (0-indexed)
    std::vector<std::vector<double>> Input, Output;
    for (const auto& row : raw_data) {
        if (row.size() >= 12) {
            Input.push_back({row[4], row[5], row[8], row[10]});  // 4 features
            Output.push_back({row[11]});  // 1 output
        }
    }

    // Create Dataset by concatenating Input and Output
    std::vector<std::vector<double>> Dataset;
    for (size_t i = 0; i < Input.size(); ++i) {
        std::vector<double> row = Input[i];
        row.push_back(Output[i][0]);
        Dataset.push_back(row);  // Each row has 5 elements (4 inputs + 1 output)
    }

    // Apply sliding window (MATLAB logic)
    // MATLAB: Input_pre = [Input_pre, Dataset(i + j - 1, :)]
    // This concatenates w rows horizontally, then removes the last element
    std::vector<std::vector<double>> input_data_rows;
    std::vector<double> output_data_vec;

    for (size_t i = 0; i < dd - w; ++i) {
        std::vector<double> Input_pre;
        for (int j = 0; j < w; ++j) {
            size_t idx = i + j;
            if (idx < Dataset.size()) {
                for (double val : Dataset[idx]) {
                    Input_pre.push_back(val);
                }
            }
        }
        // Remove the last element (which is the last output value)
        if (!Input_pre.empty()) {
            output_data_vec.push_back(Input_pre.back());
            Input_pre.pop_back();
            input_data_rows.push_back(Input_pre);
        }
    }

    std::cout << "Created " << input_data_rows.size() << " samples with sliding window." << std::endl;
    std::cout << "Each input sample has " << input_data_rows[0].size() << " features." << std::endl;

    // Transpose to match MATLAB format: [features × samples]
    int num_samples = input_data_rows.size();
    int num_features = input_data_rows[0].size();

    // Split into train and test sets
    const int numTimeStepsTrain = config["data"]["train_samples"];

    // Create tensors in [features, samples] format like MATLAB
    torch::Tensor input_train = torch::zeros({num_features, numTimeStepsTrain});
    torch::Tensor output_train = torch::zeros({numTimeStepsTrain});

    int train_limit = std::min(numTimeStepsTrain, num_samples);
    #pragma omp parallel for
    for (int i = 0; i < train_limit; ++i) {
        for (int j = 0; j < num_features; ++j) {
            input_train[j][i] = input_data_rows[i][j];
        }
        output_train[i] = output_data_vec[i];
    }

    int num_test = num_samples - numTimeStepsTrain;
    torch::Tensor input_test = torch::zeros({num_features, num_test});
    torch::Tensor output_test = torch::zeros({num_test});

    #pragma omp parallel for
    for (int i = 0; i < num_test; ++i) {
        for (int j = 0; j < num_features; ++j) {
            input_test[j][i] = input_data_rows[numTimeStepsTrain + i][j];
        }
        output_test[i] = output_data_vec[numTimeStepsTrain + i];
    }

    std::cout << "Train samples: " << numTimeStepsTrain << std::endl;
    std::cout << "Test samples: " << num_test << std::endl;
    std::cout << "Input features: " << num_features << std::endl;

    // Move tensors to device (GPU or CPU)
    input_train = input_train.to(device);
    output_train = output_train.to(device);
    input_test = input_test.to(device);
    output_test = output_test.to(device);

    // Normalize data (MATLAB's mapminmax)
    MinMaxScaler scaler;
    scaler.fit(input_train, output_train);
    torch::Tensor inputn = scaler.transform_X(input_train);
    torch::Tensor outputn = scaler.transform_Y(output_train);

    std::cout << "\nNormalized input range: [" << inputn.min().item<double>()
              << ", " << inputn.max().item<double>() << "]" << std::endl;
    std::cout << "Normalized output range: [" << outputn.min().item<double>()
              << ", " << outputn.max().item<double>() << "]" << std::endl;

    // Training loop with early stopping
    const int max_iterations = config["training"]["max_iterations"];
    const double target_r2 = config["training"]["target_r2"];

    std::shared_ptr<FeedForwardNet> best_net = nullptr;
    double best_r2 = -1e9;

    // Open training log file
    std::ofstream train_log("training_log.csv");
    train_log << "iteration,epoch,loss,learning_rate,optimizer\n";

    for (int pp = 0; pp < max_iterations; ++pp) {
        std::cout << "\n=== Training Iteration " << (pp + 1) << " ===" << std::endl;

        // Create neural network from config
        int hidden1 = config["model"]["hidden_layer1"];
        int hidden2 = config["model"]["hidden_layer2"];
        auto net = std::make_shared<FeedForwardNet>(num_features, hidden1, hidden2, 1);

        // Move model to device (GPU or CPU)
        net->to(device);

        // Prepare training data: transpose to [samples, features] for PyTorch
        torch::Tensor X_train = inputn.transpose(0, 1);  // [samples, features]
        torch::Tensor Y_train = outputn.unsqueeze(1);     // [samples, 1]

        // Training parameters from config
        const int epochs = config["training"]["epochs"];
        const double goal = config["training"]["goal_loss"];
        const int print_interval = config["training"]["print_interval"];

        // Create optimizer based on config
        std::string optimizer_type = config["optimizer"]["type"];
        std::cout << "Using optimizer: " << optimizer_type << std::endl;

        net->train();

        if (optimizer_type == "lbfgs") {
            // LBFGS optimizer
            auto lbfgs_config = config["optimizer"]["lbfgs"];
            double learning_rate = lbfgs_config["learning_rate"].get<double>();
            torch::optim::LBFGS optimizer(
                net->parameters(),
                torch::optim::LBFGSOptions(learning_rate)
                    .max_iter(lbfgs_config["max_iter"].get<size_t>())
                    .max_eval(lbfgs_config["max_eval"].get<size_t>())
                    .tolerance_grad(lbfgs_config["tolerance_grad"].get<double>())
                    .tolerance_change(lbfgs_config["tolerance_change"].get<double>())
                    .history_size(lbfgs_config["history_size"].get<size_t>())
            );

            for (int epoch = 0; epoch < epochs; ++epoch) {
                auto closure = [&]() -> torch::Tensor {
                    optimizer.zero_grad();
                    torch::Tensor output = net->forward(X_train);
                    torch::Tensor loss = torch::mse_loss(output, Y_train);
                    loss.backward();
                    // Gradient clipping to prevent exploding gradients
                    torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
                    return loss;
                };

                torch::Tensor loss = optimizer.step(closure);

                // Log training metrics
                train_log << (pp + 1) << "," << (epoch + 1) << ","
                         << loss.item<double>() << "," << learning_rate << ","
                         << optimizer_type << "\n";

                if ((epoch + 1) % print_interval == 0) {
                    std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss.item<double>() << std::endl;
                }

                if (loss.item<double>() < goal) {
                    std::cout << "Reached goal at epoch " << (epoch + 1) << std::endl;
                    break;
                }
            }
        } else if (optimizer_type == "rmsprop") {
            // RMSprop optimizer
            auto rmsprop_config = config["optimizer"]["rmsprop"];
            torch::optim::RMSprop optimizer(
                net->parameters(),
                torch::optim::RMSpropOptions(rmsprop_config["learning_rate"].get<double>())
                    .alpha(rmsprop_config["alpha"].get<double>())
                    .eps(rmsprop_config["eps"].get<double>())
                    .weight_decay(rmsprop_config["weight_decay"].get<double>())
                    .momentum(rmsprop_config["momentum"].get<double>())
                    .centered(rmsprop_config["centered"].get<bool>())
            );

            for (int epoch = 0; epoch < epochs; ++epoch) {
                optimizer.zero_grad();
                torch::Tensor output = net->forward(X_train);
                torch::Tensor loss = torch::mse_loss(output, Y_train);
                loss.backward();
                // Gradient clipping to prevent exploding gradients
                torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
                optimizer.step();

                if ((epoch + 1) % print_interval == 0) {
                    std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss.item<double>() << std::endl;
                }

                if (loss.item<double>() < goal) {
                    std::cout << "Reached goal at epoch " << (epoch + 1) << std::endl;
                    break;
                }
            }
        } else if (optimizer_type == "adam") {
            // Adam optimizer
            auto adam_config = config["optimizer"]["adam"];
            torch::optim::Adam optimizer(
                net->parameters(),
                torch::optim::AdamOptions(adam_config["learning_rate"].get<double>())
                    .betas({adam_config["beta1"].get<double>(), adam_config["beta2"].get<double>()})
                    .eps(adam_config["eps"].get<double>())
                    .weight_decay(adam_config["weight_decay"].get<double>())
            );

            for (int epoch = 0; epoch < epochs; ++epoch) {
                optimizer.zero_grad();
                torch::Tensor output = net->forward(X_train);
                torch::Tensor loss = torch::mse_loss(output, Y_train);
                loss.backward();
                // Gradient clipping to prevent exploding gradients
                torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
                optimizer.step();

                // Log training metrics
                train_log << (pp + 1) << "," << (epoch + 1) << ","
                         << loss.item<double>()  << ","
                         << optimizer_type << "\n";
                if ((epoch + 1) % print_interval == 0) {
                    std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss.item<double>() << std::endl;
                }

                if (loss.item<double>() < goal) {
                    std::cout << "Reached goal at epoch " << (epoch + 1) << std::endl;
                    break;
                }
            }
        } else if (optimizer_type == "adamw") {
            // AdamW optimizer - improved Adam with decoupled weight decay
            auto adamw_config = config["optimizer"]["adamw"];
            torch::optim::AdamW optimizer(
                net->parameters(),
                torch::optim::AdamWOptions(adamw_config["learning_rate"].get<double>())
                    .betas(std::make_tuple(adamw_config["beta1"].get<double>(), adamw_config["beta2"].get<double>()))
                    .eps(adamw_config["eps"].get<double>())
                    .weight_decay(adamw_config["weight_decay"].get<double>())
            );

            for (int epoch = 0; epoch < epochs; ++epoch) {
                optimizer.zero_grad();
                torch::Tensor output = net->forward(X_train);
                torch::Tensor loss = torch::mse_loss(output, Y_train);
                loss.backward();
                // Gradient clipping to prevent exploding gradients
                torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
                optimizer.step();
                // Log training metrics
                train_log << (pp + 1) << "," << (epoch + 1) << ","
                         << loss.item<double>()  << ","
                         << optimizer_type << "\n";
                if ((epoch + 1) % print_interval == 0) {
                    std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss.item<double>() << std::endl;
                }

                if (loss.item<double>() < goal) {
                    std::cout << "Reached goal at epoch " << (epoch + 1) << std::endl;
                    break;
                }
            }
        } else {
            std::cerr << "Unknown optimizer type: " << optimizer_type << std::endl;
            return 1;
        }

        // Log training metrics
        train_log.close();

        // Validation - MATLAB iterative prediction logic
        std::cout << "\n--- Validation ---" << std::endl;
        net->eval();
        torch::NoGradGuard no_grad;

        int kk = num_test;
        std::vector<double> data_pre;

        // Start with first test input
        torch::Tensor input = input_test.index({torch::indexing::Slice(), 0}).clone();  // [features]

        for (int n = 0; n < kk; ++n) {
            // Normalize input
            torch::Tensor inputn_new = scaler.transform_X(input);

            // Predict (reshape to [1, features])
            torch::Tensor an = net->forward(inputn_new.unsqueeze(0));

            // Denormalize output
            torch::Tensor BPoutput = scaler.inverse_transform_Y(an);
            double prediction = BPoutput[0][0].item<double>();

            data_pre.push_back(prediction);

            // Update input for next prediction (MATLAB: input(5*w-5) = output)
            if (n < kk - 1) {
                input = input_test.index({torch::indexing::Slice(), n + 1}).clone();
                // Update the (5*w-5)th element (0-indexed: 5*5-5-1 = 19)
                int update_idx = 5 * w - 5 - 1;  // MATLAB is 1-indexed
                if (update_idx < num_features) {
                    input[update_idx] = prediction;
                }
            }
        }

        // Calculate metrics
        torch::Tensor YPred = torch::zeros({kk}, device);
        for (int i = 0; i < kk; ++i) {
            YPred[i] = data_pre[i];
        }

        double RMSE = calculateRMSE(output_test, YPred);
        double R2 = calculateRSquared(output_test, YPred);

        std::cout << "RMSE: " << RMSE << std::endl;
        std::cout << "R-squared value : " << R2 << std::endl;

        // Check if this is the best model
        if (R2 > best_r2) {
            best_r2 = R2;
            best_net = net;
        }

        // Early stopping
        if (R2 > target_r2) {
            std::cout << "\nTarget R² reached! Saving model..." << std::endl;
            torch::save(net, "best_model.pt");
            break;
        }
    }

    // Final validation with best model
    if (best_net) {
        std::cout << "\n=== Final Validation with Best Model ===" << std::endl;
        best_net->eval();
        torch::NoGradGuard no_grad;

        int kk = num_test;
        std::vector<double> data_pre;

        torch::Tensor input = input_test.index({torch::indexing::Slice(), 0}).clone();

        for (int n = 0; n < kk; ++n) {
            torch::Tensor inputn_new = scaler.transform_X(input);
            torch::Tensor an = best_net->forward(inputn_new.unsqueeze(0));
            torch::Tensor BPoutput = scaler.inverse_transform_Y(an);
            double prediction = BPoutput[0][0].item<double>();

            data_pre.push_back(prediction);

            if (n < kk - 1) {
                input = input_test.index({torch::indexing::Slice(), n + 1}).clone();
                int update_idx = 5 * w - 5 - 1;
                if (update_idx < num_features) {
                    input[update_idx] = prediction;
                }
            }
        }

        torch::Tensor YPred = torch::zeros({kk}, device);
        for (int i = 0; i < kk; ++i) {
            YPred[i] = data_pre[i];
        }

        double RMSE = calculateRMSE(output_test, YPred);
        double R2 = calculateRSquared(output_test, YPred);

        std::cout << "Final RMSE: " << RMSE << std::endl;
        std::cout << "Final R²: " << R2 << std::endl;

        // Save predictions
        std::ofstream outfile("predictions.csv");
        outfile << "YTest,YPred,Error\n";
        for (int i = 0; i < kk; ++i) {
            double y_test = output_test[i].item<double>();
            double y_pred = YPred[i].item<double>();
            double error = y_test - y_pred;
            outfile << y_test << "," << y_pred << "," << error << "\n";
        }
        outfile.close();
        std::cout << "\nPredictions saved to predictions.csv" << std::endl;
    }

    std::cout << "\nTraining and validation completed!" << std::endl;
    return 0;
}

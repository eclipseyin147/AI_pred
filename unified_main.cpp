#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <omp.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include "ffn_manager.h"
#include "sedm_manager.h"
#include "faultdiag_manager.h"

using json = nlohmann::json;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [config_file.json]\n\n";
    std::cout << "Unified executable for TJU-Torch project.\n";
    std::cout << "Modes: ffn, sedm, faultdiag (set in config file)\n\n";
    std::cout << "If no config file is provided, 'unified_config.json' will be used.\n";
    std::cout << std::endl;
}

json load_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open config file '" << config_file << "'." << std::endl;
        std::cerr << "Using default configuration." << std::endl;

        return json{
            {"mode", "ffn"},
            {"ffn", {
                {"input_data_path", "Data_V13_40kW.txt"},
                {"output_model_path", "ffn_best_model.pt"},
                {"output_predictions_path", "predictions.csv"},
                {"output_training_log_path", "training_log.csv"},
                {"hidden_layers", 2},
                {"hidden_layer_neurons", {50, 50}},
                {"learning_rate", 0.001},
                {"epochs", 15000},
                {"batch_size", 32},
                {"optimizer_type", "adamw"},
                {"goal_loss", 2e-5},
                {"max_iterations", 1000},
                {"target_r2", 0.85},
                {"print_interval", 200},
                {"window_size", 5},
                {"train_samples", 300},
                {"num_rows", 900}
            }},
            {"sedm", {
                {"input_data_path", "Data_V13_40kW.txt"},
                {"model_path", "sedm_best_model.pt"},
                {"output_predictions_path", "hybrid_predictions.csv"},
                {"hidden_layers", 2},
                {"hidden_layer_neurons", {50, 50}},
                {"learning_rate", 1.0},
                {"epochs", 1000},
                {"batch_size", 32},
                {"optimizer_type", "lbfgs"},
                {"goal_loss", 1e-10},
                {"window_size", 5},
                {"train_samples", 300},
                {"num_rows", 900},
                {"rr", 4.0}
            }},
            {"faultdiag", {
                {"submode", "tcn"},
                {"input_mat_path", "ALL_Traindata1.mat"},
                {"output_model_path", "fault_best_model.pt"},
                {"hidden_layers", 2},
                {"hidden_layer_neurons", {64, 48}},
                {"learning_rate", 0.001},
                {"epochs", 100},
                {"batch_size", 26},
                {"optimizer", "adam"},
                {"use_gpu", false},
                {"data_var", "AXTrain3"},
                {"label_var", "AYTrain"},
                {"val_data_var", "AXTest3"},
                {"val_label_var", "AYTest"},
                {"train_split", 0.8},
                {"validation_frequency", 10},
                {"cnn_filter_size", 2},
                {"cnn_num_filters", 32},
                {"tcn_num_blocks", 4},
                {"tcn_num_filters", 64},
                {"tcn_filter_size", 3},
                {"tcn_dropout", 0.005}
            }}
        };
    }

    json config;
    file >> config;
    file.close();
    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "=== TJU-Torch Unified Executable ===" << std::endl;

    // Parse command line
    std::string config_file = "unified_config.json";
    if (argc > 1) {
        if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        config_file = argv[1];
    }

    std::cout << "Loading configuration from: " << config_file << std::endl;
    json config = load_config(config_file);

    std::string mode = config.value("mode", "ffn");
    std::cout << "\nRunning mode: " << mode << std::endl;

    // Set OpenMP threads
    int num_threads = omp_get_max_threads();
    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    if (omp_env != nullptr) {
        num_threads = std::atoi(omp_env);
    }
    omp_set_num_threads(num_threads);
    std::cout << "OpenMP enabled with " << num_threads << " threads." << std::endl;

    // Configure PyTorch threading
    torch::set_num_threads(num_threads);
    torch::set_num_interop_threads(num_threads);
    std::cout << "PyTorch configured to use " << num_threads << " threads." << std::endl;

    // Dispatch to appropriate manager
    if (mode == "ffn") {
        if (!config.contains("ffn")) {
            std::cerr << "Error: Config missing 'ffn' section." << std::endl;
            return 1;
        }
        tju_torch::FFNManager manager;
        manager.run(config["ffn"]);
    } else if (mode == "sedm") {
        if (!config.contains("sedm")) {
            std::cerr << "Error: Config missing 'sedm' section." << std::endl;
            return 1;
        }
        tju_torch::SEDMManager manager;
        manager.run(config["sedm"]);
    } else if (mode == "faultdiag") {
        if (!config.contains("faultdiag")) {
            std::cerr << "Error: Config missing 'faultdiag' section." << std::endl;
            return 1;
        }
        tju_torch::FaultDiagManager manager;
        manager.run(config["faultdiag"]);
    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'" << std::endl;
        std::cerr << "Valid modes: 'ffn', 'sedm', 'faultdiag'" << std::endl;
        return 1;
    }

    std::cout << "\n=== Execution Completed ===" << std::endl;
    return 0;
}

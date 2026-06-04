#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <omp.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include "sedm_manager.h"
#include "faultdiag_manager.h"

using json = nlohmann::json;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [config_file.json]\n\n";
    std::cout << "Unified executable for TJU-Torch project.\n";
    std::cout << "Modes: battery_lifespan, faultdiag (set in config file)\n";
    std::cout << "  battery_lifespan submodes: train, predict\n\n";
    std::cout << "If no config file is provided, 'unified_config.json' will be used.\n";
    std::cout << std::endl;
}

json load_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open config file '" << config_file << "'." << std::endl;
        std::cerr << "Using default configuration." << std::endl;

        return json{
            {"mode", "battery_lifespan"},
            {"battery_lifespan", {
                {"submode", "train"},
                {"input_data_path", "Data_V13_40kW.txt"},
                {"model_path", "battery_best_model.pt"},
                {"output_predictions_path", "battery_predictions.csv"},
                {"output_training_log_path", "battery_training_log.csv"},
                {"control_file_path", "control.json"},
                {"status_file_path", "status.json"},
                {"hidden_layers", 2},
                {"hidden_layer_neurons", {50, 50}},
                {"learning_rate", 1.0},
                {"epochs", 1000},
                {"batch_size", 32},
                {"optimizer_type", "lbfgs"},
                {"optimizer", {
                    {"lbfgs", {
                        {"learning_rate", 1.0},
                        {"max_iter", 20},
                        {"max_eval", 25},
                        {"tolerance_grad", 1e-7},
                        {"tolerance_change", 1e-9},
                        {"history_size", 100}
                    }},
                    {"adamw", {
                        {"learning_rate", 0.001},
                        {"beta1", 0.9},
                        {"beta2", 0.999},
                        {"eps", 1e-8},
                        {"weight_decay", 0.001}
                    }}
                }},
                {"normalization", {
                    {"enabled", true},
                    {"method", "minmax_neg1_1"}
                }},
                {"goal_loss", 1e-10},
                {"max_iterations", 1000},
                {"target_r2", 0.85},
                {"print_interval", 200},
                {"window_size", 5},
                {"train_samples", 300},
                {"num_rows", 900},
                {"rr", 4.0},
                {"input_columns", {4, 5, 8, 10}},
                {"output_column", 11},
                {"time_column", 0},
                {"nn", 300},
                {"A_cell", 0.019},
                {"t_MEM", 0.000015},
                {"t_CLc", 0.000015},
                {"t_MPLc", 0.00003},
                {"t_GDLc", 0.00018},
                {"t_CHc", 0.00044},
                {"POR_CLc", 0.455},
                {"POR_MPLc", 0.4},
                {"POR_GDLc", 0.6},
                {"Alpha_a", 0.8},
                {"Alpha_c", 0.2},
                {"j_ref_a", 10.0},
                {"j_ref_c", 0.00001},
                {"K_c_ini", 100.0},
                {"b_leak", 0.001},
                {"b_ECSA", -0.0002},
                {"b_ion", 0.0002},
                {"b_R", 1e-8},
                {"b_D", 0.1},
                {"b_B", 0.00001}
            }},
            {"faultdiag", {
                {"submode", "tcn"},
                {"input_mat_path", "ALL_Traindata1.mat"},
                {"output_model_path", "fault_best_model.pt"},
                {"control_file_path", "control.json"},
                {"status_file_path", "status.json"},
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
                {"normalization", {
                    {"enabled", true},
                    {"method", "rescale_symmetric"}
                }},
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

    std::string mode = config.value("mode", "battery_lifespan");
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
    if (mode == "battery_lifespan") {
        if (!config.contains("battery_lifespan")) {
            std::cerr << "Error: Config missing 'battery_lifespan' section." << std::endl;
            return 1;
        }
        std::string submode = config["battery_lifespan"].value("submode", "predict");
        std::cout << "Battery lifespan submode: " << submode << std::endl;
        tju_torch::BatteryLifespanManager manager;
        manager.run(config["battery_lifespan"]);
    } else if (mode == "faultdiag") {
        if (!config.contains("faultdiag")) {
            std::cerr << "Error: Config missing 'faultdiag' section." << std::endl;
            return 1;
        }
        tju_torch::FaultDiagManager manager;
        manager.run(config["faultdiag"]);
    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'" << std::endl;
        std::cerr << "Valid modes: 'battery_lifespan', 'faultdiag'" << std::endl;
        return 1;
    }

    std::cout << "\n=== Execution Completed ===" << std::endl;
    return 0;
}

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

namespace {

constexpr const char* kDefaultConfigFile = "unified_config.json";
constexpr const char* kDefaultMode = "battery_lifespan";
constexpr const char* kDefaultLifespanSubmode = "train";

struct CliOptions {
    std::string config_path = kDefaultConfigFile;
    std::string mode;
    std::string submode;
    bool show_help = false;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [--mode <mode>] [--submode <submode>] [--config <path>]\n\n";
    std::cout << "Unified executable for TJU-Torch project.\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --mode <mode>       Tool type: battery_lifespan | faultdiag\n";
    std::cout << "                          (default: " << kDefaultMode << ")\n";
    std::cout << "  -s, --submode <submode> Lifespan task (battery_lifespan only): train | predict\n";
    std::cout << "                          (default: " << kDefaultLifespanSubmode << ")\n";
    std::cout << "  -c, --config <path>     Config file path (default: ./" << kDefaultConfigFile << ")\n";
    std::cout << "  -h, --help              Show this help\n\n";
    std::cout << "Run task (mode/submode) comes from CLI with built-in defaults, not from JSON.\n";
    std::cout << "JSON holds persistent parameters (paths, hyperparameters, physics settings).\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --mode battery_lifespan --submode train\n";
    std::cout << "  " << program_name << " --mode battery_lifespan --submode predict\n";
    std::cout << "  " << program_name << " --mode faultdiag\n";
    std::cout << std::endl;
}

bool parse_cli(int argc, char* argv[], CliOptions& cli) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cli.show_help = true;
            return true;
        }
        if (arg == "--mode" || arg == "-m") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value." << std::endl;
                return false;
            }
            cli.mode = argv[++i];
            continue;
        }
        if (arg == "--submode" || arg == "-s") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value." << std::endl;
                return false;
            }
            cli.submode = argv[++i];
            continue;
        }
        if (arg == "--config" || arg == "-c") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value." << std::endl;
                return false;
            }
            cli.config_path = argv[++i];
            continue;
        }
        std::cerr << "Error: Unknown argument '" << arg << "'." << std::endl;
        return false;
    }
    return true;
}

bool load_config(const std::string& config_file, json& config) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Error: Config file not found: " << config_file << std::endl;
        return false;
    }
    try {
        file >> config;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to parse config file '" << config_file << "': " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool is_valid_mode(const std::string& mode) {
    return mode == "battery_lifespan" || mode == "faultdiag";
}

bool is_valid_lifespan_submode(const std::string& submode) {
    return submode == "train" || submode == "predict";
}

} // namespace

int main(int argc, char* argv[]) {
    std::cout << "=== TJU-Torch Unified Executable ===" << std::endl;

    CliOptions cli;
    if (!parse_cli(argc, argv, cli)) {
        print_usage(argv[0]);
        return 1;
    }
    if (cli.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    const std::string mode = cli.mode.empty() ? kDefaultMode : cli.mode;
    if (!is_valid_mode(mode)) {
        std::cerr << "Error: Unknown mode '" << mode << "'." << std::endl;
        std::cerr << "Valid modes: 'battery_lifespan', 'faultdiag'" << std::endl;
        return 1;
    }

    std::string submode;
    if (mode == "battery_lifespan") {
        submode = cli.submode.empty() ? kDefaultLifespanSubmode : cli.submode;
        if (!is_valid_lifespan_submode(submode)) {
            std::cerr << "Error: Unknown submode '" << submode << "' for battery_lifespan." << std::endl;
            std::cerr << "Valid submodes: 'train', 'predict'" << std::endl;
            return 1;
        }
    } else if (!cli.submode.empty()) {
        std::cerr << "Warning: --submode is ignored when mode=faultdiag (uses faultdiag.submode from JSON)."
                  << std::endl;
    }

    const std::string config_file = cli.config_path;
    std::cout << "Loading configuration from: " << config_file << std::endl;

    json config;
    if (!load_config(config_file, config)) {
        return 1;
    }

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
        std::cout << "Battery lifespan submode: " << submode << std::endl;
        tju_torch::BatteryLifespanManager manager;
        manager.run(config["battery_lifespan"], submode);
    } else if (mode == "faultdiag") {
        if (!config.contains("faultdiag")) {
            std::cerr << "Error: Config missing 'faultdiag' section." << std::endl;
            return 1;
        }
        tju_torch::FaultDiagManager manager;
        manager.run(config["faultdiag"]);
    }

    std::cout << "\n=== Execution Completed ===" << std::endl;
    return 0;
}

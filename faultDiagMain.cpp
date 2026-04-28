//
// Created by siqi on 12/10/2025.
//

#include "faultDiagnosis.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
using namespace lifespanPred;
using json = nlohmann::json;

void print_usage() {
    std::cout << "Usage: faultDiagnosis [config_file.json]\n\n";
    std::cout << "If no config file is provided, 'faultDiag_config.json' will be used.\n\n";
    std::cout << "Example config file:\n";
    std::cout << "{\n";
    std::cout << "  \"mode\": \"tcn\",  // \"cnn\", \"tcn\", or \"test\"\n";
    std::cout << "  \"data\": {\n";
    std::cout << "    \"mat_file\": \"ALL_Traindata1.mat\",\n";
    std::cout << "    \"data_var\": \"AXTrain3\",\n";
    std::cout << "    \"label_var\": \"AYTrain\",\n";
    std::cout << "    \"val_data_var\": \"AXTest3\",\n";
    std::cout << "    \"val_label_var\": \"AYTest\",\n";
    std::cout << "    \"train_split\": 0.8\n";
    std::cout << "  },\n";
    std::cout << "  \"training\": {\n";
    std::cout << "    \"epochs\": 100,\n";
    std::cout << "    \"batch_size\": 26,\n";
    std::cout << "    \"learning_rate\": 0.001,\n";
    std::cout << "    \"validation_frequency\": 10,\n";
    std::cout << "    \"optimizer\": \"adam\",\n";
    std::cout << "    \"use_gpu\": false\n";
    std::cout << "  },\n";
    std::cout << "  \"model\": {\n";
    std::cout << "    \"cnn\": {\n";
    std::cout << "      \"filter_size\": 2,\n";
    std::cout << "      \"num_filters\": 32\n";
    std::cout << "    },\n";
    std::cout << "    \"tcn\": {\n";
    std::cout << "      \"num_blocks\": 4,\n";
    std::cout << "      \"num_filters\": 64,\n";
    std::cout << "      \"filter_size\": 3,\n";
    std::cout << "      \"dropout\": 0.005\n";
    std::cout << "    }\n";
    std::cout << "  },\n";
    std::cout << "  \"output\": {\n";
    std::cout << "    \"model_save_path\": \"best_model.pt\"\n";
    std::cout << "  }\n";
    std::cout << "}\n";
    std::cout << std::endl;
}

json load_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file '" << config_file << "'." << std::endl;
        std::cerr << "Using default configuration." << std::endl;

        // Return default configuration
        return json{
            {"mode", "test"},
            {"data", {
                {"mat_file", ""},
                {"data_var", "XTrain"},
                {"label_var", "YTrain"},
                {"val_data_var", ""},
                {"val_label_var", ""},
                {"train_split", 0.8}
            }},
            {"training", {
                {"epochs", 100},
                {"batch_size", 26},
                {"learning_rate", 0.001},
                {"validation_frequency", 10},
                {"optimizer", "adam"},
                {"use_gpu", false}
            }},
            {"model", {
                {"cnn", {
                    {"filter_size", 2},
                    {"num_filters", 32}
                }},
                {"tcn", {
                    {"num_blocks", 4},
                    {"num_filters", 64},
                    {"filter_size", 3},
                    {"dropout", 0.005}
                }}
            }},
            {"output", {
                {"model_save_path", "best_model.pt"}
            }}
        };
    }

    json config;
    file >> config;
    file.close();
    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Lifespan Prediction with LibTorch ===" << std::endl;

    // Load configuration file
    auto cwd = std::filesystem::current_path();
    std::string config_file = cwd.u8string()+"/faultDiag_config.json";
    if (argc > 1) {
        if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
            print_usage();
            return 0;
        }
        config_file = argv[1];
    }


    std::cout << "Loading configuration from: " << config_file << std::endl;
    json config = load_config(config_file);

    // Print loaded configuration
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << config.dump(2) << std::endl << std::endl;

    // Extract configuration values
    std::string mode = config["mode"];
    std::string mat_file = config["data"]["mat_file"];
    std::string data_var = config["data"]["data_var"];
    std::string label_var = config["data"]["label_var"];

    // Check for separate validation data variables (optional)
    std::string val_data_var = "";
    std::string val_label_var = "";
    if (config["data"].contains("val_data_var")) {
        val_data_var = config["data"]["val_data_var"];
    }
    if (config["data"].contains("val_label_var")) {
        val_label_var = config["data"]["val_label_var"];
    }

    double train_split = config["data"]["train_split"];

    int epochs = config["training"]["epochs"];
    int batch_size = config["training"]["batch_size"];
    double learning_rate = config["training"]["learning_rate"];
    int validation_frequency = config["training"]["validation_frequency"];
    std::string optimizer = config["training"]["optimizer"];
    bool use_gpu = config["training"]["use_gpu"];

    std::string model_save_path = config["output"]["model_save_path"];

    // Set device
    torch::Device device(torch::kCUDA);
    if (use_gpu && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using GPU for training" << std::endl;
    } else {
        std::cout << "Using CPU for training" << std::endl;
    }

    // Test mode with synthetic data
    if (mode == "test") {
        std::cout << "\n=== Testing with Synthetic Data ===" << std::endl;

        // Create synthetic sequences
        std::vector<torch::Tensor> train_sequences;
        std::vector<int64_t> train_labels;

        int num_features = 8;
        int num_classes = 4;
        int num_samples = 100;

        for (int i = 0; i < num_samples; ++i) {
            int seq_length = 50 + (i % 50);  // Variable length: 50-99
            torch::Tensor seq = torch::randn({num_features, seq_length});
            train_sequences.push_back(seq);
            train_labels.push_back(i % num_classes);
        }

        std::cout << "Generated " << num_samples << " synthetic sequences" << std::endl;

        // Get model parameters from config
        int cnn_filter_size = config["model"]["cnn"]["filter_size"];
        int cnn_num_filters = config["model"]["cnn"]["num_filters"];

        int tcn_num_blocks = config["model"]["tcn"]["num_blocks"];
        int tcn_num_filters = config["model"]["tcn"]["num_filters"];
        int tcn_filter_size = config["model"]["tcn"]["filter_size"];
        double tcn_dropout = config["model"]["tcn"]["dropout"];

        // Train 1D CNN
        std::cout << "\n--- Training 1D CNN ---" << std::endl;
        Conv1DNet cnn_model(num_features, num_classes,
                           cnn_filter_size, cnn_num_filters);

        TrainingConfig cnn_config;
        cnn_config.max_epochs = epochs;
        cnn_config.mini_batch_size = batch_size;
        cnn_config.learning_rate = learning_rate;
        cnn_config.validation_frequency = validation_frequency;
        cnn_config.optimizer_type = optimizer;
        cnn_config.device = device;
        cnn_config.normalize_input = true;  // CNN: rescale-symmetric normalization (matches MATLAB)
        cnn_config.model_save_path = "cnn_test_model.pt";

        SequenceTrainer<Conv1DNet> cnn_trainer(cnn_model, cnn_config);
        cnn_trainer.train(train_sequences, train_labels);

        // Train TCN
        std::cout << "\n--- Training TCN ---" << std::endl;
        TCNNet tcn_model(num_features, num_classes,
                        tcn_num_blocks, tcn_num_filters,
                        tcn_filter_size, tcn_dropout);

        TrainingConfig tcn_config;
        tcn_config.max_epochs = epochs;
        tcn_config.mini_batch_size = batch_size;
        tcn_config.learning_rate = learning_rate;
        tcn_config.validation_frequency = validation_frequency;
        tcn_config.optimizer_type = optimizer;
        tcn_config.device = device;
        tcn_config.normalize_input = true;  // TCN: rescale-symmetric normalization
        tcn_config.model_save_path = "tcn_test_model.pt";

        SequenceTrainer<TCNNet> tcn_trainer(tcn_model, tcn_config);
        tcn_trainer.train(train_sequences, train_labels);

        std::cout << "\nTest completed successfully!" << std::endl;
        return 0;
    }

    // Check if MAT file is provided
    if (mat_file.empty()) {
        std::cerr << "Error: --mat-file is required for training modes" << std::endl;
        print_usage();
        return 1;
    }

    // Load data from MAT file
    std::cout << "\nLoading training data from: " << mat_file << std::endl;
    std::cout << "  Data variable: " << data_var << std::endl;
    std::cout << "  Label variable: " << label_var << std::endl;

    auto [sequences, labels] = load_mat_data(mat_file, data_var, label_var);

    if (sequences.empty() || labels.empty()) {
        std::cerr << "Error: Failed to load training data from MAT file" << std::endl;
        return 1;
    }

    // Determine number of features and classes
    int64_t num_features = sequences[0].size(0);
    int64_t num_classes = *std::max_element(labels.begin(), labels.end()) + 1;

    std::cout << "Number of features: " << num_features << std::endl;
    std::cout << "Number of classes: " << num_classes << std::endl;
    std::cout << "Number of training samples: " << sequences.size() << std::endl;

    // Prepare train and validation data
    std::vector<torch::Tensor> train_sequences;
    std::vector<int64_t> train_labels;
    std::vector<torch::Tensor> val_sequences;
    std::vector<int64_t> val_labels;

    // Check if separate validation data is specified
    bool use_separate_validation = !val_data_var.empty() && !val_label_var.empty();

    if (use_separate_validation) {
        // Load separate validation data from MAT file
        std::cout << "\nLoading validation data from: " << mat_file << std::endl;
        std::cout << "  Validation data variable: " << val_data_var << std::endl;
        std::cout << "  Validation label variable: " << val_label_var << std::endl;

        auto [val_seqs, val_lbls] = load_mat_data(mat_file, val_data_var, val_label_var);

        if (val_seqs.empty() || val_lbls.empty()) {
            std::cerr << "Warning: Failed to load validation data. Falling back to train/val split." << std::endl;
            // Fall back to splitting
            size_t train_size = static_cast<size_t>(sequences.size() * train_split);
            train_sequences = std::vector<torch::Tensor>(sequences.begin(), sequences.begin() + train_size);
            train_labels = std::vector<int64_t>(labels.begin(), labels.begin() + train_size);
            val_sequences = std::vector<torch::Tensor>(sequences.begin() + train_size, sequences.end());
            val_labels = std::vector<int64_t>(labels.begin() + train_size, labels.end());
        } else {
            // Use all training data and separate validation data
            train_sequences = sequences;
            train_labels = labels;
            val_sequences = val_seqs;
            val_labels = val_lbls;
            std::cout << "Number of validation samples: " << val_sequences.size() << std::endl;
        }
    } else {
        // Split into train and validation
        std::cout << "\nNo separate validation data specified. Splitting training data..." << std::endl;
        size_t train_size = static_cast<size_t>(sequences.size() * train_split);

        train_sequences = std::vector<torch::Tensor>(sequences.begin(), sequences.begin() + train_size);
        train_labels = std::vector<int64_t>(labels.begin(), labels.begin() + train_size);

        val_sequences = std::vector<torch::Tensor>(sequences.begin() + train_size, sequences.end());
        val_labels = std::vector<int64_t>(labels.begin() + train_size, labels.end());
    }

    std::cout << "Final training samples: " << train_sequences.size() << std::endl;
    std::cout << "Final validation samples: " << val_sequences.size() << std::endl;

    // Configure training
    TrainingConfig train_config;
    train_config.max_epochs = epochs;
    train_config.mini_batch_size = batch_size;
    train_config.learning_rate = learning_rate;
    train_config.validation_frequency = validation_frequency;
    train_config.device = device;
    train_config.optimizer_type = optimizer;
    // Set normalization based on mode
    // Both CNN and TCN use rescale-symmetric normalization [-1, 1]
    train_config.normalize_input = true;
    // Get model parameters from config
    int cnn_filter_size = config["model"]["cnn"]["filter_size"];
    int cnn_num_filters = config["model"]["cnn"]["num_filters"];

    int tcn_num_blocks = config["model"]["tcn"]["num_blocks"];
    int tcn_num_filters = config["model"]["tcn"]["num_filters"];
    int tcn_filter_size = config["model"]["tcn"]["filter_size"];
    double tcn_dropout = config["model"]["tcn"]["dropout"];

    // Train based on mode
    if (mode == "cnn") {
        std::cout << "\n=== Training 1D CNN Model ===" << std::endl;
        std::cout << "Model config: filter_size=" << cnn_filter_size
                  << ", num_filters=" << cnn_num_filters << std::endl;

        Conv1DNet model(num_features, num_classes,
                       cnn_filter_size, cnn_num_filters);
        train_config.model_save_path = model_save_path;

        SequenceTrainer<Conv1DNet> trainer(model, train_config);
        trainer.train(train_sequences, train_labels, val_sequences, val_labels);

        // Final evaluation
        std::cout << "\n=== Final Evaluation ===" << std::endl;
        double train_acc = trainer.evaluate(train_sequences, train_labels);
        double val_acc = trainer.evaluate(val_sequences, val_labels);

        std::cout << "Training Accuracy: " << (train_acc * 100) << "%" << std::endl;
        std::cout << "Validation Accuracy: " << (val_acc * 100) << "%" << std::endl;

        // Create confusion matrix
        auto predictions = trainer.predict(val_sequences);
        torch::Tensor pred_tensor = torch::tensor(predictions);
        torch::Tensor label_tensor = torch::tensor(val_labels);

        torch::Tensor confusion = create_confusion_matrix(pred_tensor, label_tensor, num_classes);

        std::vector<std::string> class_names;
        for (int64_t i = 0; i < num_classes; ++i) {
            class_names.push_back("Class " + std::to_string(i + 1));
        }
        print_confusion_matrix(confusion, class_names);

    } else if (mode == "tcn") {
        std::cout << "\n=== Training TCN Model ===" << std::endl;
        std::cout << "Model config: num_blocks=" << tcn_num_blocks
                  << ", num_filters=" << tcn_num_filters
                  << ", filter_size=" << tcn_filter_size
                  << ", dropout=" << tcn_dropout << std::endl;

        TCNNet model(num_features, num_classes,
                    tcn_num_blocks, tcn_num_filters,
                    tcn_filter_size, tcn_dropout);
        train_config.model_save_path = model_save_path;

        SequenceTrainer<TCNNet> trainer(model, train_config);
        trainer.train(train_sequences, train_labels, val_sequences, val_labels);

        // Final evaluation
        std::cout << "\n=== Final Evaluation ===" << std::endl;
        double train_acc = trainer.evaluate(train_sequences, train_labels);
        double val_acc = trainer.evaluate(val_sequences, val_labels);

        std::cout << "Training Accuracy: " << (train_acc * 100) << "%" << std::endl;
        std::cout << "Validation Accuracy: " << (val_acc * 100) << "%" << std::endl;

        // Create confusion matrix
        auto predictions = trainer.predict(val_sequences);
        torch::Tensor pred_tensor = torch::tensor(predictions);
        torch::Tensor label_tensor = torch::tensor(val_labels);

        torch::Tensor confusion = create_confusion_matrix(pred_tensor, label_tensor, num_classes);

        std::vector<std::string> class_names;
        for (int64_t i = 0; i < num_classes; ++i) {
            class_names.push_back("Class " + std::to_string(i + 1));
        }
        print_confusion_matrix(confusion, class_names);

    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'" << std::endl;
        std::cerr << "Valid modes: 'cnn', 'tcn', 'test'" << std::endl;
        return 1;
    }

    std::cout << "\n=== Training Completed ===" << std::endl;
    return 0;
}

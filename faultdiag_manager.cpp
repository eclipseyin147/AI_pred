#include "faultdiag_manager.h"
#include "faultDiagnosis.h"
#include "training_controller.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

namespace tju_torch {

using namespace lifespanPred;
using json = nlohmann::json;

void FaultDiagManager::run(const nlohmann::json& config) {
    std::cout << "=== Fault Diagnosis with LibTorch ===" << std::endl;

    // File paths must be explicitly configured in JSON (no hard-coded defaults)
    if (!config.contains("input_mat_path")) {
        std::cerr << "Error: config missing 'input_mat_path'" << std::endl;
        return;
    }
    if (!config.contains("output_model_path")) {
        std::cerr << "Error: config missing 'output_model_path'" << std::endl;
        return;
    }
    if (!config.contains("control_file_path")) {
        std::cerr << "Error: config missing 'control_file_path'" << std::endl;
        return;
    }
    if (!config.contains("status_file_path")) {
        std::cerr << "Error: config missing 'status_file_path'" << std::endl;
        return;
    }

    std::string submode = config.value("submode", "tcn");
    std::string mat_file = config["input_mat_path"];
    std::string data_var = config.value("data_var", "XTrain");
    std::string label_var = config.value("label_var", "YTrain");
    std::string val_data_var = config.value("val_data_var", "");
    std::string val_label_var = config.value("val_label_var", "");
    double train_split = config.value("train_split", 0.8);

    int epochs = config.value("epochs", 100);
    int batch_size = config.value("batch_size", 26);
    double learning_rate = config.value("learning_rate", 0.001);
    int validation_frequency = config.value("validation_frequency", 10);
    std::string optimizer = config.value("optimizer", "adam");
    bool use_gpu = config.value("use_gpu", false);
    std::string model_save_path = config["output_model_path"];
    std::string control_file = config["control_file_path"];
    std::string status_file = config["status_file_path"];

    int cnn_filter_size = config.value("cnn_filter_size", 2);
    int cnn_num_filters = config.value("cnn_num_filters", 32);
    int tcn_num_blocks = config.value("tcn_num_blocks", 4);
    int tcn_num_filters = config.value("tcn_num_filters", 64);
    int tcn_filter_size = config.value("tcn_filter_size", 3);
    double tcn_dropout = config.value("tcn_dropout", 0.005);

    // Normalization method for fault diagnosis
    SequenceNormMethod norm_method = SequenceNormMethod::RESCALE_SYMMETRIC;
    bool norm_enabled = true;
    if (config.contains("normalization")) {
        norm_enabled = config["normalization"].value("enabled", true);
        if (norm_enabled) {
            std::string method_str = config["normalization"].value("method", "rescale_symmetric");
            if (method_str == "minmax_0_1") norm_method = SequenceNormMethod::MINMAX_0_1;
            else if (method_str == "z_score") norm_method = SequenceNormMethod::Z_SCORE;
            else norm_method = SequenceNormMethod::RESCALE_SYMMETRIC;
        }
    }

    // Training controller
    TrainingController controller(control_file, status_file);

    // Device selection
    torch::Device device(torch::kCPU);
    if (use_gpu && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using GPU for training" << std::endl;
    } else {
        std::cout << "Using CPU for training" << std::endl;
    }

    // Test mode with synthetic data
    if (submode == "test") {
        std::cout << "\n=== Testing with Synthetic Data ===" << std::endl;

        std::vector<torch::Tensor> train_sequences;
        std::vector<int64_t> train_labels;

        int num_features = 8;
        int num_classes = 4;
        int num_samples = 100;

        for (int i = 0; i < num_samples; ++i) {
            int seq_length = 50 + (i % 50);
            torch::Tensor seq = torch::randn({num_features, seq_length});
            train_sequences.push_back(seq);
            train_labels.push_back(i % num_classes);
        }

        std::cout << "Generated " << num_samples << " synthetic sequences" << std::endl;

        // Train 1D CNN
        std::cout << "\n--- Training 1D CNN ---" << std::endl;
        Conv1DNet cnn_model(num_features, num_classes, cnn_filter_size, cnn_num_filters);

        TrainingConfig cnn_config;
        cnn_config.max_epochs = epochs;
        cnn_config.mini_batch_size = batch_size;
        cnn_config.learning_rate = learning_rate;
        cnn_config.validation_frequency = validation_frequency;
        cnn_config.optimizer_type = optimizer;
        cnn_config.device = device;
        cnn_config.normalize_input = true;
        cnn_config.normalization_method = norm_method;
        cnn_config.model_save_path = "cnn_test_model.pt";
        cnn_config.control_file_path = control_file;
        cnn_config.status_file_path = status_file;

        SequenceTrainer<Conv1DNet> cnn_trainer(cnn_model, cnn_config);
        cnn_trainer.train(train_sequences, train_labels, {}, {}, &controller);

        // Train TCN
        std::cout << "\n--- Training TCN ---" << std::endl;
        TCNNet tcn_model(num_features, num_classes, tcn_num_blocks, tcn_num_filters,
                        tcn_filter_size, tcn_dropout);

        TrainingConfig tcn_config;
        tcn_config.max_epochs = epochs;
        tcn_config.mini_batch_size = batch_size;
        tcn_config.learning_rate = learning_rate;
        tcn_config.validation_frequency = validation_frequency;
        tcn_config.optimizer_type = optimizer;
        tcn_config.device = device;
        tcn_config.normalize_input = true;
        tcn_config.normalization_method = norm_method;
        tcn_config.model_save_path = "tcn_test_model.pt";
        tcn_config.control_file_path = control_file;
        tcn_config.status_file_path = status_file;

        SequenceTrainer<TCNNet> tcn_trainer(tcn_model, tcn_config);
        tcn_trainer.train(train_sequences, train_labels, {}, {}, &controller);

        std::cout << "\nTest completed successfully!" << std::endl;
        return;
    }

    // Check if MAT file is provided
    if (mat_file.empty()) {
        std::cerr << "Error: input_mat_path is required for training modes" << std::endl;
        return;
    }

    // Load data from MAT file
    std::cout << "\nLoading training data from: " << mat_file << std::endl;
    std::cout << "  Data variable: " << data_var << std::endl;
    std::cout << "  Label variable: " << label_var << std::endl;

    auto [sequences, labels] = load_mat_data(mat_file, data_var, label_var);

    if (sequences.empty() || labels.empty()) {
        std::cerr << "Error: Failed to load training data from MAT file" << std::endl;
        return;
    }

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

    bool use_separate_validation = !val_data_var.empty() && !val_label_var.empty();

    if (use_separate_validation) {
        std::cout << "\nLoading validation data from: " << mat_file << std::endl;
        std::cout << "  Validation data variable: " << val_data_var << std::endl;
        std::cout << "  Validation label variable: " << val_label_var << std::endl;

        auto [val_seqs, val_lbls] = load_mat_data(mat_file, val_data_var, val_label_var);

        if (val_seqs.empty() || val_lbls.empty()) {
            std::cerr << "Warning: Failed to load validation data. Falling back to train/val split." << std::endl;
            size_t train_size = static_cast<size_t>(sequences.size() * train_split);
            train_sequences = std::vector<torch::Tensor>(sequences.begin(), sequences.begin() + train_size);
            train_labels = std::vector<int64_t>(labels.begin(), labels.begin() + train_size);
            val_sequences = std::vector<torch::Tensor>(sequences.begin() + train_size, sequences.end());
            val_labels = std::vector<int64_t>(labels.begin() + train_size, labels.end());
        } else {
            train_sequences = sequences;
            train_labels = labels;
            val_sequences = val_seqs;
            val_labels = val_lbls;
            std::cout << "Number of validation samples: " << val_sequences.size() << std::endl;
        }
    } else {
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
    train_config.normalize_input = true;
    train_config.normalization_method = norm_method;
    train_config.model_save_path = model_save_path;
    train_config.control_file_path = control_file;
    train_config.status_file_path = status_file;

    // Check for restart command
    std::string initial_cmd = controller.read_command();
    if (initial_cmd == "restart") {
        controller.acknowledge_command();
        controller.clear_checkpoint(model_save_path + ".checkpoint.json",
                                    model_save_path + ".checkpoint.pt");
        std::cout << "Restart command received: starting fresh training." << std::endl;
    }

    // Train based on submode
    if (submode == "cnn") {
        std::cout << "\n=== Training 1D CNN Model ===" << std::endl;
        std::cout << "Model config: filter_size=" << cnn_filter_size
                  << ", num_filters=" << cnn_num_filters << std::endl;

        Conv1DNet model(num_features, num_classes, cnn_filter_size, cnn_num_filters);

        SequenceTrainer<Conv1DNet> trainer(model, train_config);
        trainer.train(train_sequences, train_labels, val_sequences, val_labels, &controller);

        // Final evaluation
        std::cout << "\n=== Final Evaluation ===" << std::endl;
        double train_acc = trainer.evaluate(train_sequences, train_labels);
        double val_acc = trainer.evaluate(val_sequences, val_labels);

        std::cout << "Training Accuracy: " << (train_acc * 100) << "%" << std::endl;
        std::cout << "Validation Accuracy: " << (val_acc * 100) << "%" << std::endl;

        auto predictions = trainer.predict(val_sequences);
        torch::Tensor pred_tensor = torch::tensor(predictions);
        torch::Tensor label_tensor = torch::tensor(val_labels);

        torch::Tensor confusion = create_confusion_matrix(pred_tensor, label_tensor, num_classes);

        std::vector<std::string> class_names;
        for (int64_t i = 0; i < num_classes; ++i) {
            class_names.push_back("Class " + std::to_string(i + 1));
        }
        print_confusion_matrix(confusion, class_names);

    } else if (submode == "tcn") {
        std::cout << "\n=== Training TCN Model ===" << std::endl;
        std::cout << "Model config: num_blocks=" << tcn_num_blocks
                  << ", num_filters=" << tcn_num_filters
                  << ", filter_size=" << tcn_filter_size
                  << ", dropout=" << tcn_dropout << std::endl;

        TCNNet model(num_features, num_classes, tcn_num_blocks, tcn_num_filters,
                    tcn_filter_size, tcn_dropout);

        SequenceTrainer<TCNNet> trainer(model, train_config);
        trainer.train(train_sequences, train_labels, val_sequences, val_labels, &controller);

        // Final evaluation
        std::cout << "\n=== Final Evaluation ===" << std::endl;
        double train_acc = trainer.evaluate(train_sequences, train_labels);
        double val_acc = trainer.evaluate(val_sequences, val_labels);

        std::cout << "Training Accuracy: " << (train_acc * 100) << "%" << std::endl;
        std::cout << "Validation Accuracy: " << (val_acc * 100) << "%" << std::endl;

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
        std::cerr << "Error: Unknown submode '" << submode << "'" << std::endl;
        std::cerr << "Valid submodes: 'cnn', 'tcn', 'test'" << std::endl;
        return;
    }

    std::cout << "\n=== Training Completed ===" << std::endl;
}

} // namespace tju_torch

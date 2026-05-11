//
// Created by siqi on 12/10/2025.
//

#ifndef TJU_TORCH_LIFESPANNPRED_H
#define TJU_TORCH_LIFESPANNPRED_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <torch/nn/modules/dropout.h>
#include "training_controller.h"
namespace lifespanPred {

// ============================================================================
// 1D Convolutional Neural Network (similar to m1DCNN.m)
// ============================================================================
struct Conv1DNetImpl : torch::nn::Module {
    torch::nn::Conv1d conv1{nullptr}, conv2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Linear fc{nullptr};
    int64_t num_classes;

    Conv1DNetImpl(int64_t num_features, int64_t num_classes,
                  int64_t filter_size = 2, int64_t num_filters = 32);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Conv1DNet);

// ============================================================================
// TCNBlock with Residual Connection
// ============================================================================
struct TCNBlockImpl : torch::nn::Module {
    torch::nn::Conv1d conv1{nullptr};
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Conv1d skip_conv{nullptr};  // For first block to match dimensions
    int64_t dilation_factor;
    double dropout_factor;  // Store dropout rate for manual spatial dropout implementation
    bool is_first_block;

    TCNBlockImpl(int64_t num_features, int64_t num_filters,
                 int64_t filter_size, int64_t dilation_factor,
                 double dropout_factor, bool is_first_block = false);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(TCNBlock);

// ============================================================================
// Temporal Convolutional Network (similar to myselfTCN.m)
// ============================================================================
struct TCNNetImpl : torch::nn::Module {
    torch::nn::ModuleList blocks;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    int64_t num_features;
    int64_t num_filters;

    TCNNetImpl(int64_t num_features, int64_t num_classes,
               int64_t num_blocks = 4, int64_t num_filters = 64,
               int64_t filter_size = 3, double dropout_factor = 0.005);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(TCNNet);

// ============================================================================
// Data Normalization (MATLAB's rescale-symmetric: [-1, 1])
// ============================================================================
enum class SequenceNormMethod {
    RESCALE_SYMMETRIC,  // [-1, 1] (original MATLAB behavior)
    MINMAX_0_1,         // [0, 1]
    Z_SCORE             // (x - mean) / std
};

class SequenceNormalizer {
public:
    SequenceNormMethod method = SequenceNormMethod::RESCALE_SYMMETRIC;
    torch::Tensor mean, std_dev, min_val, max_val;
    bool fitted = false;

    explicit SequenceNormalizer(SequenceNormMethod m = SequenceNormMethod::RESCALE_SYMMETRIC)
        : method(m) {}

    // Fit normalizer on training data
    void fit(const std::vector<torch::Tensor>& sequences);

    // Transform sequences based on selected method
    std::vector<torch::Tensor> transform(const std::vector<torch::Tensor>& sequences);

    // Inverse transform
    std::vector<torch::Tensor> inverse_transform(const std::vector<torch::Tensor>& sequences);

    // Save/load normalizer parameters
    void save(const std::string& filename);
    void load(const std::string& filename);
};

// ============================================================================
// Data Loading and Preprocessing
// ============================================================================
class SequenceDataset : public torch::data::Dataset<SequenceDataset> {
public:
    std::vector<torch::Tensor> sequences;
    std::vector<int64_t> labels;

    SequenceDataset() = default;
    SequenceDataset(std::vector<torch::Tensor> seqs, std::vector<int64_t> lbls)
        : sequences(std::move(seqs)), labels(std::move(lbls)) {}

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};

// ============================================================================
// Training Configuration
// ============================================================================
struct TrainingConfig {
    int64_t max_epochs = 200;
    int64_t mini_batch_size = 26;
    double learning_rate = 0.001;
    int64_t validation_frequency = 10;
    bool verbose = true;
    std::string optimizer_type = "adam";  // adam, sgd, rmsprop
    torch::Device device = torch::kCPU;

    // Data normalization
    bool normalize_input = false;  // Set to true for TCN, false for CNN
    SequenceNormMethod normalization_method = SequenceNormMethod::RESCALE_SYMMETRIC;

    // Adam optimizer parameters
    double adam_beta1 = 0.9;
    double adam_beta2 = 0.999;
    double adam_eps = 1e-8;
    double weight_decay = 0.0;

    // Model save path
    std::string model_save_path = "best_model.pt";

    // Control files for Qt frontend IPC
    std::string control_file_path = "control.json";
    std::string status_file_path = "status.json";
};

// ============================================================================
// Trainer Class (Template)
// ============================================================================
template<typename ModelType>
class SequenceTrainer {
public:
    TrainingConfig config;
    ModelType model;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    SequenceNormalizer normalizer;

    SequenceTrainer(ModelType model_ptr, TrainingConfig cfg)
        : model(model_ptr), config(cfg), normalizer(cfg.normalization_method) {

        // Move model to device
        model->to(config.device);

        // Create optimizer
        if (config.optimizer_type == "adam") {
            optimizer = std::make_unique<torch::optim::Adam>(
                model->parameters(),
                torch::optim::AdamOptions(config.learning_rate)
                    .betas(std::make_tuple(config.adam_beta1, config.adam_beta2))
                    .eps(config.adam_eps)
                    .weight_decay(config.weight_decay)
            );
        }
        else if (config.optimizer_type == "adam") {

        }
        else if (config.optimizer_type == "sgd") {
            optimizer = std::make_unique<torch::optim::SGD>(
                model->parameters(),
                torch::optim::SGDOptions(config.learning_rate)
            );
        } else if (config.optimizer_type == "rmsprop") {
            optimizer = std::make_unique<torch::optim::RMSprop>(
                model->parameters(),
                torch::optim::RMSpropOptions(config.learning_rate)
            );
        } else {
            throw std::runtime_error("Unknown optimizer: " + config.optimizer_type);
        }
    }

    // Train the model
    void train(const std::vector<torch::Tensor>& train_sequences,
               const std::vector<int64_t>& train_labels,
               const std::vector<torch::Tensor>& val_sequences = {},
               const std::vector<int64_t>& val_labels = {},
               tju_torch::TrainingController* controller = nullptr);

    // Evaluate the model
    double evaluate(const std::vector<torch::Tensor>& sequences,
                   const std::vector<int64_t>& labels);

    // Predict on new sequences
    std::vector<int64_t> predict(const std::vector<torch::Tensor>& sequences);

    // Save/load model
    void save_model(const std::string& path);
    void load_model(const std::string& path);
};

// ============================================================================
// Utility Functions
// ============================================================================

// Calculate accuracy
double calculate_accuracy(const torch::Tensor& predictions, const torch::Tensor& targets);

// Create confusion matrix
torch::Tensor create_confusion_matrix(const torch::Tensor& predictions,
                                      const torch::Tensor& targets,
                                      int64_t num_classes);

// Print confusion matrix
void print_confusion_matrix(const torch::Tensor& confusion_matrix,
                           const std::vector<std::string>& class_names);

// Collate function for variable-length sequences (with padding)
struct SequenceCollate {
    std::string padding_direction = "left";  // "left" or "right"

    std::vector<torch::Tensor> operator()(
        std::vector<torch::data::Example<>> examples);
};

// Load data from .mat file (placeholder - requires external library like matio)
std::pair<std::vector<torch::Tensor>, std::vector<int64_t>>
load_mat_data(const std::string& filename,
              const std::string& data_var_name,
              const std::string& label_var_name);

// ============================================================================
// Template Method Implementations
// ============================================================================

template<typename ModelType>
void SequenceTrainer<ModelType>::train(const std::vector<torch::Tensor>& train_sequences,
                                       const std::vector<int64_t>& train_labels,
                                       const std::vector<torch::Tensor>& val_sequences,
                                       const std::vector<int64_t>& val_labels,
                                       tju_torch::TrainingController* controller) {
    // Normalize data if enabled (TCN uses normalization, CNN doesn't)
    std::vector<torch::Tensor> norm_train_sequences;
    if (config.normalize_input) {
        normalizer.fit(train_sequences);
        norm_train_sequences = normalizer.transform(train_sequences);
    } else {
        norm_train_sequences = train_sequences;
    }

    // Training loop
    double best_val_acc = 0.0;

    std::cout << "Starting training for " << config.max_epochs << " epochs..." << std::endl;
    if (config.normalize_input) {
        std::string method_name = "rescale-symmetric [-1,1]";
        if (config.normalization_method == SequenceNormMethod::MINMAX_0_1) method_name = "minmax [0,1]";
        if (config.normalization_method == SequenceNormMethod::Z_SCORE) method_name = "z-score";
        std::cout << "Input normalization: ENABLED (" << method_name << ")" << std::endl;
    } else {
        std::cout << "Input normalization: DISABLED" << std::endl;
    }

    if (controller) {
        controller->update_status("faultdiag", "running", 0, static_cast<int>(config.max_epochs),
                                  0.0, 0.0, 0.0, 0.0, "Training started");
    }

    for (int64_t epoch = 0; epoch < config.max_epochs; ++epoch) {
        // Check control commands
        if (controller) {
            std::string cmd = controller->read_command();
            if (cmd == "pause") {
                controller->acknowledge_command();
                controller->update_status("faultdiag", "paused", static_cast<int>(epoch),
                                          static_cast<int>(config.max_epochs), 0.0, 0.0, 0.0, 0.0,
                                          "Paused by user");
                // Save checkpoint
                save_model(config.model_save_path + ".checkpoint.pt");
                nlohmann::json meta = {{"epoch", epoch}, {"best_val_acc", best_val_acc}};
                controller->save_checkpoint_meta(config.model_save_path + ".checkpoint.json", meta);
                std::cout << "Checkpoint saved. Waiting for resume..." << std::endl;

                std::string resume_cmd = controller->wait_for_resume();
                if (resume_cmd == "stop") {
                    controller->update_status("faultdiag", "stopped", static_cast<int>(epoch),
                                              static_cast<int>(config.max_epochs), 0.0, 0.0, 0.0, 0.0,
                                              "Stopped by user");
                    return;
                }
                if (resume_cmd == "restart") {
                    controller->clear_checkpoint(config.model_save_path + ".checkpoint.json",
                                                 config.model_save_path + ".checkpoint.pt");
                    controller->update_status("faultdiag", "running", 0,
                                              static_cast<int>(config.max_epochs), 0.0, 0.0, 0.0, 0.0,
                                              "Restarting fresh");
                    epoch = -1;
                    best_val_acc = 0.0;
                    continue;
                }
                controller->update_status("faultdiag", "running", static_cast<int>(epoch),
                                          static_cast<int>(config.max_epochs), 0.0, 0.0, 0.0, 0.0,
                                          "Resumed");
            }
            if (cmd == "stop") {
                controller->acknowledge_command();
                save_model(config.model_save_path + ".checkpoint.pt");
                nlohmann::json meta = {{"epoch", epoch}, {"best_val_acc", best_val_acc}};
                controller->save_checkpoint_meta(config.model_save_path + ".checkpoint.json", meta);
                controller->update_status("faultdiag", "stopped", static_cast<int>(epoch),
                                          static_cast<int>(config.max_epochs), 0.0, 0.0, 0.0, 0.0,
                                          "Stopped by user");
                return;
            }
            if (cmd == "restart") {
                controller->acknowledge_command();
                controller->clear_checkpoint(config.model_save_path + ".checkpoint.json",
                                             config.model_save_path + ".checkpoint.pt");
                controller->update_status("faultdiag", "running", 0,
                                          static_cast<int>(config.max_epochs), 0.0, 0.0, 0.0, 0.0,
                                          "Restarting fresh");
                epoch = -1;
                best_val_acc = 0.0;
                continue;
            }
        }

        model->train();

        double total_loss = 0.0;
        int64_t num_batches = 0;

        // Simple batch training
        for (size_t i = 0; i < train_sequences.size(); i += config.mini_batch_size) {
            size_t batch_end = std::min(i + config.mini_batch_size,
                                       train_sequences.size());

            // Prepare batch
            std::vector<torch::Tensor> batch_seqs;
            std::vector<int64_t> batch_labels;

            for (size_t j = i; j < batch_end; ++j) {
                batch_seqs.push_back(norm_train_sequences[j]);
                batch_labels.push_back(train_labels[j]);
            }

            // Pad sequences to same length
            int64_t max_len = 0;
            for (const auto& seq : batch_seqs) {
                max_len = std::max(max_len, seq.size(1));
            }

            std::vector<torch::Tensor> padded_seqs;
            for (const auto& seq : batch_seqs) {
                int64_t pad_size = max_len - seq.size(1);
                if (pad_size > 0) {
                    auto padding = torch::zeros({seq.size(0), pad_size});
                    padded_seqs.push_back(torch::cat({padding, seq}, 1));
                } else {
                    padded_seqs.push_back(seq);
                }
            }

            // Stack to batch tensor
            torch::Tensor batch_data = torch::stack(padded_seqs, 0);
            torch::Tensor batch_targets = torch::tensor(batch_labels);

            batch_data = batch_data.to(config.device);
            batch_targets = batch_targets.to(config.device);

            // Forward pass
            optimizer->zero_grad();
            torch::Tensor output = model->forward(batch_data);
            torch::Tensor loss = torch::nn::functional::cross_entropy(output, batch_targets);

            // Backward pass
            loss.backward();
            optimizer->step();

            total_loss += loss.item<double>();
            num_batches++;
        }

        double avg_loss = total_loss / num_batches;

        if (config.verbose && (epoch + 1) % config.validation_frequency == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << config.max_epochs
                     << "], Loss: " << std::fixed << std::setprecision(6) << avg_loss;

            // Validation
            if (!val_sequences.empty()) {
                double val_acc = evaluate(val_sequences, val_labels);
                std::cout << ", Val Acc: " << std::setprecision(4) << val_acc;

                // Save best model
                if (val_acc > best_val_acc) {
                    best_val_acc = val_acc;
                    save_model(config.model_save_path);
                    std::cout << " (Best model saved)";
                }
            }

            std::cout << std::endl;

            // Update controller status
            if (controller) {
                controller->update_status("faultdiag", "running", static_cast<int>(epoch + 1),
                                          static_cast<int>(config.max_epochs), avg_loss, 0.0,
                                          0.0, 0.0,
                                          "Epoch " + std::to_string(epoch + 1));
            }
        }
    }

    std::cout << "Training completed!" << std::endl;
    if (controller) {
        controller->update_status("faultdiag", "completed",
                                  static_cast<int>(config.max_epochs),
                                  static_cast<int>(config.max_epochs), 0.0,
                                  0.0, 0.0, 0.0, "Training completed");
    }
}

template<typename ModelType>
double SequenceTrainer<ModelType>::evaluate(const std::vector<torch::Tensor>& sequences,
                                           const std::vector<int64_t>& labels) {
    model->eval();
    torch::NoGradGuard no_grad;

    auto predictions = predict(sequences);

    int64_t correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
    }

    return static_cast<double>(correct) / predictions.size();
}

template<typename ModelType>
std::vector<int64_t> SequenceTrainer<ModelType>::predict(const std::vector<torch::Tensor>& sequences) {
    model->eval();
    torch::NoGradGuard no_grad;

    // Normalize if enabled
    std::vector<torch::Tensor> norm_sequences;
    if (config.normalize_input) {
        norm_sequences = normalizer.transform(sequences);
    } else {
        norm_sequences = sequences;
    }

    std::vector<int64_t> predictions;

    for (const auto& seq : norm_sequences) {
        torch::Tensor input = seq.unsqueeze(0).to(config.device);
        torch::Tensor output = model->forward(input);
        int64_t pred = output.argmax(1).item<int64_t>();
        predictions.push_back(pred);
    }

    return predictions;
}

template<typename ModelType>
void SequenceTrainer<ModelType>::save_model(const std::string& path) {
    torch::save(model, path);
    normalizer.save(path + ".normalizer");
}

template<typename ModelType>
void SequenceTrainer<ModelType>::load_model(const std::string& path) {
    torch::load(model, path);
    normalizer.load(path + ".normalizer");
}

} // namespace lifespanPred

#endif //TJU_TORCH_LIFESPANNPRED_H
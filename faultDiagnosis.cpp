//
// Created by siqi on 12/10/2025.
//

#include "faultDiagnosis.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <matio.h>

namespace lifespanPred {

// ============================================================================
// Conv1DNet Implementation
// ============================================================================
Conv1DNetImpl::Conv1DNetImpl(int64_t num_features, int64_t num_classes,
                             int64_t filter_size, int64_t num_filters)
    : num_classes(num_classes) {

    // Conv1d expects input shape: [batch, channels, sequence_length]
    // num_features is the number of channels (features per time step)

    // First conv layer
    // For "same" padding: padding = (kernel_size - 1) / 2
    // This matches MATLAB's behavior for odd kernel sizes
    // For even kernel sizes (like 2), output will be L-1 instead of L+1
    conv1 = register_module("conv1",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(num_features, num_filters, filter_size)
            .padding((filter_size - 1) / 2)));  // Correct "same" padding formula

    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({num_filters})));

    // Second conv layer
    conv2 = register_module("conv2",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(num_filters, num_filters, filter_size)
            .padding((filter_size - 1) / 2)));  // Correct "same" padding formula

    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({num_filters})));

    // Fully connected layer
    fc = register_module("fc", torch::nn::Linear(num_filters, num_classes));
}

torch::Tensor Conv1DNetImpl::forward(torch::Tensor x) {
    // Input x shape: [batch, features, sequence_length]

    // First conv block: Conv → ReLU → LayerNorm (matches MATLAB)
    x = conv1->forward(x);
    x = torch::relu(x);
    x = x.permute({0, 2, 1}).contiguous();  // [batch, seq_len, filters] - ensure contiguous for CUDA
    x = norm1->forward(x);
    x = x.permute({0, 2, 1}).contiguous();  // [batch, filters, seq_len] - ensure contiguous for CUDA

    // Second conv block: Conv → ReLU → LayerNorm (matches MATLAB)
    x = conv2->forward(x);
    x = torch::relu(x);
    x = x.permute({0, 2, 1}).contiguous();  // [batch, seq_len, filters] - ensure contiguous for CUDA
    x = norm2->forward(x);
    x = x.permute({0, 2, 1}).contiguous();  // [batch, filters, seq_len] - ensure contiguous for CUDA

    // Global max pooling
    x = std::get<0>(torch::adaptive_max_pool1d(x, 1));  // [batch, filters, 1]
    x = x.squeeze(2);  // [batch, filters]

    // Fully connected layer
    x = fc->forward(x);  // [batch, num_classes]

    return x;
}

// ============================================================================
// TCNBlock Implementation
// ============================================================================
TCNBlockImpl::  TCNBlockImpl(int64_t num_features, int64_t num_filters,
                           int64_t filter_size, int64_t dilation_factor,
                           double dropout_factor, bool is_first_block)
    : dilation_factor(dilation_factor), dropout_factor(dropout_factor), is_first_block(is_first_block) {

    // For causal convolution, we'll use padding=0 and manually pad left in forward()
    conv1 = register_module("conv1",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(
            is_first_block ? num_features : num_filters,
            num_filters, filter_size)
            .dilation(dilation_factor)
            .padding(0)));  // No automatic padding

    norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({num_filters})));

    // Note: Spatial dropout is implemented manually in forward() because:
    // 1. torch::nn::Dropout1d doesn't exist in older LibTorch versions
    // 2. Regular torch::nn::Dropout is element-wise, not channel-wise
    // 3. MATLAB's spatialDropoutLayer drops entire channels, which we replicate manually

    // Skip connection for first block
    if (is_first_block) {
        skip_conv = register_module("skip_conv",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(num_features, num_filters, 1)
                .padding(0)));
    }
}

torch::Tensor TCNBlockImpl::forward(torch::Tensor x) {
    // Save input for residual connection
    torch::Tensor residual = x;

    // Causal padding: pad LEFT only with (filter_size - 1) * dilation
    int64_t kernel_size = conv1->options.kernel_size()->at(0);
    int64_t left_pad = (kernel_size - 1) * dilation_factor;

    if (left_pad > 0) {
        // F.pad format: (left, right, top, bottom, front, back)
        // For 3D tensor [batch, channels, length], we pad the last dimension
        x = torch::nn::functional::pad(x,
            torch::nn::functional::PadFuncOptions({left_pad, 0})  // left pad only
                .mode(torch::kConstant)
                .value(0));
    }

    // Convolutional layer (now with manually added causal padding)
    x = conv1->forward(x);

    // Layer normalization
    x = x.permute({0, 2, 1}).contiguous();  // [batch, seq_len, filters] - ensure contiguous for CUDA
    x = norm->forward(x);
    x = x.permute({0, 2, 1}).contiguous();  // [batch, filters, seq_len] - ensure contiguous for CUDA

    // ReLU activation
    x = torch::relu(x);

    // Manual Spatial Dropout (matches MATLAB's spatialDropoutLayer)
    // This drops ENTIRE CHANNELS across all timesteps, not individual elements
    if (is_training() && dropout_factor > 0.0) {
        x = x.contiguous();

        // Get dimensions
        int64_t batch_size = x.size(0);
        int64_t num_channels = x.size(1);
        int64_t seq_length = x.size(2);

        // Create channel-wise dropout mask: [batch, channels, 1]
        // Each channel is either fully kept or fully dropped across all timesteps
        torch::Tensor noise = torch::rand({batch_size, num_channels, 1}, x.options());

        // Create binary mask: 1 if keep, 0 if drop
        torch::Tensor mask = (noise > dropout_factor).to(x.dtype());

        // Apply inverted dropout scaling: scale by 1/(1-p) to maintain expected value
        mask = mask / (1.0 - dropout_factor);

        // Apply mask - broadcasts [batch, channels, 1] across [batch, channels, seq_len]
        x = x * mask;
        x = x.contiguous();
    }

    // Skip connection
    if (is_first_block) {
        residual = skip_conv->forward(residual);
        // Ensure skip_conv output is contiguous for CUDA
        residual = residual.contiguous();
    }

    // Addition (residual connection)
    // Ensure both tensors are contiguous before addition for CUDA stability
    x = x.contiguous();
    residual = residual.contiguous();
    x = x + residual;

    // Ensure output is contiguous for next block
    x = x.contiguous();

    return x;
}

// ============================================================================
// TCNNet Implementation
// ============================================================================
TCNNetImpl::TCNNetImpl(int64_t num_features, int64_t num_classes,
                       int64_t num_blocks, int64_t num_filters,
                       int64_t filter_size, double dropout_factor)
    : num_features(num_features), num_filters(num_filters) {

    blocks = register_module("blocks", torch::nn::ModuleList());

    // Create TCN blocks with increasing dilation
    for (int64_t i = 0; i < num_blocks; ++i) {
        int64_t dilation = static_cast<int64_t>(std::pow(2, i));
        bool is_first = (i == 0);

        auto block = TCNBlock(num_features, num_filters, filter_size,
                             dilation, dropout_factor, is_first);
        blocks->push_back(block);
    }

    // Fully connected layers
    fc1 = register_module("fc1", torch::nn::Linear(num_filters, 48));
    fc2 = register_module("fc2", torch::nn::Linear(48, num_classes));
}

torch::Tensor TCNNetImpl::forward(torch::Tensor x) {
    // Input x shape: [batch, features, sequence_length]

    // Pass through TCN blocks
    for (const auto& block : *blocks) {
        x = block->as<TCNBlock>()->forward(x);
    }

    // Ensure tensor is contiguous before indexing (critical for CUDA)
    x = x.contiguous();

    // Take the last timestep for classification (matches MATLAB behavior)
    // MATLAB's fullyConnectedLayer automatically uses the final timestep
    // x shape: [batch, filters, seq_len] → [batch, filters]
    x = x.index({torch::indexing::Slice(), torch::indexing::Slice(), -1});

    // Fully connected layers
    x = fc1->forward(x);
    x = torch::relu(x);
    x = fc2->forward(x);  // [batch, num_classes]

    return x;
}

// ============================================================================
// SequenceNormalizer Implementation
// ============================================================================
void SequenceNormalizer::fit(const std::vector<torch::Tensor>& sequences) {
    if (sequences.empty()) return;

    std::vector<torch::Tensor> all_data;
    all_data.reserve(sequences.size());
    for (const auto& seq : sequences) {
        all_data.push_back(seq);
    }

    torch::Tensor concat_data = torch::cat(all_data, 1);  // [features, total_length]

    if (method == SequenceNormMethod::Z_SCORE) {
        mean = concat_data.mean(1);  // [features]
        std_dev = concat_data.std(1, true);
        std_dev = torch::where(std_dev == 0, torch::ones_like(std_dev), std_dev);
    } else {
        // Min-max variants
        min_val = std::get<0>(concat_data.min(1));  // [features]
        max_val = std::get<0>(concat_data.max(1));  // [features]
    }

    fitted = true;
}

std::vector<torch::Tensor> SequenceNormalizer::transform(const std::vector<torch::Tensor>& sequences) {
    if (!fitted) {
        throw std::runtime_error("Normalizer not fitted. Call fit() first.");
    }

    std::vector<torch::Tensor> normalized;
    normalized.reserve(sequences.size());
    for (const auto& seq : sequences) {
        torch::Tensor norm_seq;
        if (method == SequenceNormMethod::Z_SCORE) {
            norm_seq = (seq - mean.unsqueeze(1)) / std_dev.unsqueeze(1);
        } else {
            auto range = max_val - min_val;
            range = torch::where(range == 0, torch::ones_like(range), range);
            if (method == SequenceNormMethod::MINMAX_0_1) {
                norm_seq = (seq - min_val.unsqueeze(1)) / range.unsqueeze(1);
            } else {
                // RESCALE_SYMMETRIC: [-1, 1]
                norm_seq = 2.0 * (seq - min_val.unsqueeze(1)) / range.unsqueeze(1) - 1.0;
            }
        }
        normalized.push_back(norm_seq);
    }

    return normalized;
}

std::vector<torch::Tensor> SequenceNormalizer::inverse_transform(const std::vector<torch::Tensor>& sequences) {
    if (!fitted) {
        throw std::runtime_error("Normalizer not fitted. Call fit() first.");
    }

    std::vector<torch::Tensor> denormalized;
    denormalized.reserve(sequences.size());
    for (const auto& seq : sequences) {
        torch::Tensor denorm_seq;
        if (method == SequenceNormMethod::Z_SCORE) {
            denorm_seq = seq * std_dev.unsqueeze(1) + mean.unsqueeze(1);
        } else {
            auto range = max_val - min_val;
            range = torch::where(range == 0, torch::ones_like(range), range);
            if (method == SequenceNormMethod::MINMAX_0_1) {
                denorm_seq = seq * range.unsqueeze(1) + min_val.unsqueeze(1);
            } else {
                // RESCALE_SYMMETRIC: [-1, 1]
                denorm_seq = (seq + 1.0) * range.unsqueeze(1) / 2.0 + min_val.unsqueeze(1);
            }
        }
        denormalized.push_back(denorm_seq);
    }

    return denormalized;
}

void SequenceNormalizer::save(const std::string& filename) {
    torch::save({min_val, max_val, mean, std_dev}, filename);
}

void SequenceNormalizer::load(const std::string& filename) {
    std::vector<torch::Tensor> tensors;
    torch::load(tensors, filename);
    if (tensors.size() >= 4) {
        min_val = tensors[0];
        max_val = tensors[1];
        mean = tensors[2];
        std_dev = tensors[3];
    } else {
        min_val = tensors[0];
        max_val = tensors[1];
    }
    fitted = true;
}

// ============================================================================
// SequenceDataset Implementation
// ============================================================================
torch::data::Example<> SequenceDataset::get(size_t index) {
    return {sequences[index], torch::tensor(static_cast<int64_t>(labels[index]))};
}

torch::optional<size_t> SequenceDataset::size() const {
    return sequences.size();
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================
double calculate_accuracy(const torch::Tensor& predictions, const torch::Tensor& targets) {
    auto correct = (predictions == targets).sum();
    return correct.item<double>() / predictions.size(0);
}

torch::Tensor create_confusion_matrix(const torch::Tensor& predictions,
                                      const torch::Tensor& targets,
                                      int64_t num_classes) {
    torch::Tensor confusion = torch::zeros({num_classes, num_classes}, torch::kInt64);

    for (int64_t i = 0; i < predictions.size(0); ++i) {
        int64_t pred = predictions[i].item<int64_t>();
        int64_t target = targets[i].item<int64_t>();
        confusion[target][pred] += 1;
    }

    return confusion;
}

void print_confusion_matrix(const torch::Tensor& confusion_matrix,
                           const std::vector<std::string>& class_names) {
    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << std::setw(10) << " ";

    for (const auto& name : class_names) {
        std::cout << std::setw(10) << name;
    }
    std::cout << std::endl;

    for (size_t i = 0; i < class_names.size(); ++i) {
        std::cout << std::setw(10) << class_names[i];
        for (size_t j = 0; j < class_names.size(); ++j) {
            std::cout << std::setw(10) << confusion_matrix[i][j].item<int64_t>();
        }
        std::cout << std::endl;
    }
}

std::vector<torch::Tensor> SequenceCollate::operator()(
    std::vector<torch::data::Example<>> examples) {

    std::vector<torch::Tensor> sequences;
    std::vector<torch::Tensor> labels;

    for (auto& example : examples) {
        sequences.push_back(example.data);
        labels.push_back(example.target);
    }

    // Find max sequence length
    int64_t max_len = 0;
    for (const auto& seq : sequences) {
        max_len = std::max(max_len, seq.size(1));
    }

    // Pad sequences
    std::vector<torch::Tensor> padded_seqs;
    for (const auto& seq : sequences) {
        int64_t pad_size = max_len - seq.size(1);
        if (pad_size > 0) {
            auto padding = torch::zeros({seq.size(0), pad_size});

            if (padding_direction == "left") {
                padded_seqs.push_back(torch::cat({padding, seq}, 1));
            } else {
                padded_seqs.push_back(torch::cat({seq, padding}, 1));
            }
        } else {
            padded_seqs.push_back(seq);
        }
    }

    return {torch::stack(padded_seqs, 0), torch::stack(labels, 0)};
}

std::pair<std::vector<torch::Tensor>, std::vector<int64_t>>
load_mat_data(const std::string& filename,
              const std::string& data_var_name,
              const std::string& label_var_name) {
    std::vector<torch::Tensor> sequences;
    std::vector<int64_t> labels;

    // Open MAT file
    mat_t *matfp = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
    if (matfp == nullptr) {
        std::cerr << "Error opening MAT file: " << filename << std::endl;
        return {sequences, labels};
    }

    // Read data variable (sequences)
    matvar_t *data_var = Mat_VarRead(matfp, data_var_name.c_str());
    if (data_var == nullptr) {
        std::cerr << "Error: Variable '" << data_var_name << "' not found in MAT file" << std::endl;
        Mat_Close(matfp);
        return {sequences, labels};
    }

    // Read label variable
    matvar_t *label_var = Mat_VarRead(matfp, label_var_name.c_str());
    if (label_var == nullptr) {
        std::cerr << "Error: Variable '" << label_var_name << "' not found in MAT file" << std::endl;
        Mat_VarFree(data_var);
        Mat_Close(matfp);
        return {sequences, labels};
    }

    // Process data variable (expecting cell array of sequences)
    if (data_var->class_type == MAT_C_CELL) {
        // Cell array of sequences
        size_t num_sequences = data_var->dims[0] * data_var->dims[1];
        matvar_t **cells = (matvar_t **)data_var->data;

        for (size_t i = 0; i < num_sequences; ++i) {
            matvar_t *cell = cells[i];
            if (cell == nullptr) continue;

            // Convert cell data to tensor
            if (cell->data_type == MAT_T_DOUBLE) {
                double *data = (double *)cell->data;
                size_t num_features = cell->dims[0];
                size_t seq_length = cell->dims[1];

                // Create tensor [features, seq_length]
                torch::Tensor tensor = torch::zeros({static_cast<int64_t>(num_features),
                                                     static_cast<int64_t>(seq_length)});

                for (size_t f = 0; f < num_features; ++f) {
                    for (size_t t = 0; t < seq_length; ++t) {
                        // MATLAB is column-major, so index is: t * num_features + f
                        tensor[f][t] = data[t * num_features + f];
                    }
                }

                sequences.push_back(tensor);
            } else if (cell->data_type == MAT_T_SINGLE) {
                float *data = (float *)cell->data;
                size_t num_features = cell->dims[0];
                size_t seq_length = cell->dims[1];

                torch::Tensor tensor = torch::zeros({static_cast<int64_t>(num_features),
                                                     static_cast<int64_t>(seq_length)});

                for (size_t f = 0; f < num_features; ++f) {
                    for (size_t t = 0; t < seq_length; ++t) {
                        tensor[f][t] = data[t * num_features + f];
                    }
                }

                sequences.push_back(tensor);
            }
        }
    } else if (data_var->data_type == MAT_T_DOUBLE || data_var->data_type == MAT_T_SINGLE) {
        // Matrix format - could be 2D or 3D
        int ndims = data_var->rank;

        if (ndims == 3) {
            // 3D matrix: [features, seq_length, num_sequences]
            size_t num_features = data_var->dims[0];
            size_t seq_length = data_var->dims[1];
            size_t num_sequences = data_var->dims[2];

            std::cout << "Processing 3D matrix: [" << num_features << ", "
                      << seq_length << ", " << num_sequences << "]" << std::endl;

            if (data_var->data_type == MAT_T_DOUBLE) {
                double *data = (double *)data_var->data;

                for (size_t s = 0; s < num_sequences; ++s) {
                    torch::Tensor tensor = torch::zeros({static_cast<int64_t>(num_features),
                                                         static_cast<int64_t>(seq_length)});

                    for (size_t f = 0; f < num_features; ++f) {
                        for (size_t t = 0; t < seq_length; ++t) {
                            // MATLAB 3D indexing (column-major): data[f + num_features * (t + seq_length * s)]
                            size_t idx = f + num_features * (t + seq_length * s);
                            tensor[f][t] = data[idx];
                        }
                    }
                    sequences.push_back(tensor);
                }
            } else if (data_var->data_type == MAT_T_SINGLE) {
                float *data = (float *)data_var->data;

                for (size_t s = 0; s < num_sequences; ++s) {
                    torch::Tensor tensor = torch::zeros({static_cast<int64_t>(num_features),
                                                         static_cast<int64_t>(seq_length)});

                    for (size_t f = 0; f < num_features; ++f) {
                        for (size_t t = 0; t < seq_length; ++t) {
                            size_t idx = f + num_features * (t + seq_length * s);
                            tensor[f][t] = data[idx];
                        }
                    }
                    sequences.push_back(tensor);
                }
            }
        } else if (ndims == 2) {
            // 2D matrix: [features, num_samples]
            // Each column is a separate sample - treat each as a sequence of length 1
            size_t num_features = data_var->dims[0];
            size_t num_samples = data_var->dims[1];

            std::cout << "Processing 2D matrix: [" << num_features << ", " << num_samples << "]" << std::endl;
            std::cout << "Treating each column as a separate sequence (seq_length=1)." << std::endl;

            if (data_var->data_type == MAT_T_DOUBLE) {
                double *data = (double *)data_var->data;

                for (size_t s = 0; s < num_samples; ++s) {
                    // Each column is a sample: create tensor [features, 1]
                    torch::Tensor tensor = torch::zeros({static_cast<int64_t>(num_features), 1});

                    for (size_t f = 0; f < num_features; ++f) {
                        // MATLAB column-major: column s is at data[f + num_features * s]
                        tensor[f][0] = data[f + num_features * s];
                    }
                    sequences.push_back(tensor);
                }
            } else if (data_var->data_type == MAT_T_SINGLE) {
                float *data = (float *)data_var->data;

                for (size_t s = 0; s < num_samples; ++s) {
                    torch::Tensor tensor = torch::zeros({static_cast<int64_t>(num_features), 1});

                    for (size_t f = 0; f < num_features; ++f) {
                        tensor[f][0] = data[f + num_features * s];
                    }
                    sequences.push_back(tensor);
                }
            }
        } else {
            std::cerr << "Warning: Unsupported matrix dimensions: " << ndims << "D" << std::endl;
        }
    }

    // Process label variable
    if (label_var->data_type == MAT_T_DOUBLE) {
        double *label_data = (double *)label_var->data;
        size_t num_labels = label_var->dims[0] * label_var->dims[1];

        for (size_t i = 0; i < num_labels; ++i) {
            // Convert to 0-based indexing (MATLAB uses 1-based)
            labels.push_back(static_cast<int64_t>(label_data[i]) - 1);
        }
    } else if (label_var->data_type == MAT_T_SINGLE) {
        float *label_data = (float *)label_var->data;
        size_t num_labels = label_var->dims[0] * label_var->dims[1];

        for (size_t i = 0; i < num_labels; ++i) {
            labels.push_back(static_cast<int64_t>(label_data[i]) - 1);
        }
    } else if (label_var->data_type == MAT_T_INT32 || label_var->data_type == MAT_T_UINT32) {
        int32_t *label_data = (int32_t *)label_var->data;
        size_t num_labels = label_var->dims[0] * label_var->dims[1];

        for (size_t i = 0; i < num_labels; ++i) {
            labels.push_back(static_cast<int64_t>(label_data[i]) - 1);
        }
    }

    // Cleanup
    Mat_VarFree(data_var);
    Mat_VarFree(label_var);
    Mat_Close(matfp);

    std::cout << "Loaded " << sequences.size() << " sequences and "
              << labels.size() << " labels from MAT file." << std::endl;

    return {sequences, labels};
}

} // namespace lifespanPred
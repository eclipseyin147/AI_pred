#include "ffn_manager.h"
#include "common_ffn.h"
#include "training_controller.h"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <omp.h>

namespace tju_torch {

void FFNManager::run(const nlohmann::json& config) {
    std::cout << std::fixed << std::setprecision(6);

    // Read paths and parameters
    // All file paths must be explicitly configured in JSON (no hard-coded defaults)
    if (!config.contains("input_data_path")) {
        std::cerr << "Error: config missing 'input_data_path'" << std::endl;
        return;
    }
    if (!config.contains("output_model_path")) {
        std::cerr << "Error: config missing 'output_model_path'" << std::endl;
        return;
    }
    if (!config.contains("output_predictions_path")) {
        std::cerr << "Error: config missing 'output_predictions_path'" << std::endl;
        return;
    }
    if (!config.contains("output_training_log_path")) {
        std::cerr << "Error: config missing 'output_training_log_path'" << std::endl;
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

    std::string data_file = config["input_data_path"];
    std::string output_model_path = config["output_model_path"];
    std::string output_predictions_path = config["output_predictions_path"];
    std::string output_training_log_path = config["output_training_log_path"];
    std::string control_file = config["control_file_path"];
    std::string status_file = config["status_file_path"];

    int num_rows = config.value("num_rows", 900);
    int window_size = config.value("window_size", 5);
    int numTimeStepsTrain = config.value("train_samples", 300);

    // Read column configuration for input/output selection
    std::vector<int> input_columns;
    if (config.contains("input_columns") && config["input_columns"].is_array()) {
        for (const auto& item : config["input_columns"]) {
            input_columns.push_back(item.get<int>());
        }
    }
    if (input_columns.empty()) {
        input_columns = {4, 5, 8, 10};
    }
    int output_column = config.value("output_column", 11);

    // Determine required minimum columns from the data file
    int max_col = output_column;
    for (int col : input_columns) {
        if (col > max_col) max_col = col;
    }
    int min_cols = max_col + 1;

    int max_iterations = config.value("max_iterations", 10);
    double target_r2 = config.value("target_r2", 0.8);
    int epochs = config.value("epochs", 1000);
    double goal_loss = config.value("goal_loss", 1e-10);
    int print_interval = config.value("print_interval", 100);

    std::string optimizer_type = config.value("optimizer_type", "lbfgs");

    // hidden_layer_neurons
    std::vector<int64_t> hidden_neurons;
    if (config.contains("hidden_layer_neurons") && config["hidden_layer_neurons"].is_array()) {
        for (const auto& item : config["hidden_layer_neurons"]) {
            hidden_neurons.push_back(item.get<int64_t>());
        }
    }
    if (hidden_neurons.empty()) {
        hidden_neurons = {50, 50};
    }

    // Normalization
    NormalizationMethod norm_method = NormalizationMethod::MINMAX_NEG1_1;
    bool norm_enabled = true;
    if (config.contains("normalization")) {
        norm_enabled = config["normalization"].value("enabled", true);
        if (norm_enabled) {
            norm_method = parse_normalization_method(config["normalization"].value("method", "minmax_neg1_1"));
        } else {
            norm_method = NormalizationMethod::NONE;
        }
    }

    // Training controller
    TrainingController controller(control_file, status_file);
    std::string checkpoint_model = output_model_path + ".checkpoint.pt";
    std::string checkpoint_meta = output_model_path + ".checkpoint.json";

    // Device selection
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    } else {
        std::cout << "CUDA not available. Training on CPU with OpenMP acceleration." << std::endl;
    }

    // Load data
    std::cout << "\nLoading data from " << data_file << "..." << std::endl;
    auto raw_data = readDataFile(data_file, num_rows);
    if (raw_data.empty()) {
        std::cerr << "Error: No data loaded!" << std::endl;
        controller.update_status("ffn", "stopped", 0, epochs, 0.0, 0.0, 0.0, 0.0, "Data load failed");
        return;
    }
    std::cout << "Loaded " << raw_data.size() << " rows of data." << std::endl;

    const int w = window_size;
    const size_t dd = raw_data.size();

    // Extract Input and Output columns
    std::vector<std::vector<double>> Input, Output;
    for (const auto& row : raw_data) {
        if (row.size() >= static_cast<size_t>(min_cols)) {
            std::vector<double> in_row;
            for (int col : input_columns) {
                in_row.push_back(row[col]);
            }
            Input.push_back(in_row);
            Output.push_back({row[output_column]});
        }
    }

    std::vector<std::vector<double>> Dataset;
    for (size_t i = 0; i < Input.size(); ++i) {
        std::vector<double> row = Input[i];
        row.push_back(Output[i][0]);
        Dataset.push_back(row);
    }

    // Apply sliding window
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
        if (!Input_pre.empty()) {
            output_data_vec.push_back(Input_pre.back());
            Input_pre.pop_back();
            input_data_rows.push_back(Input_pre);
        }
    }

    std::cout << "Created " << input_data_rows.size() << " samples with sliding window." << std::endl;
    std::cout << "Each input sample has " << input_data_rows[0].size() << " features." << std::endl;

    int num_samples = static_cast<int>(input_data_rows.size());
    int num_features = static_cast<int>(input_data_rows[0].size());

    int train_limit = std::min(numTimeStepsTrain, num_samples);
    int num_test = num_samples - train_limit;

    torch::Tensor input_train = torch::zeros({num_features, train_limit});
    torch::Tensor output_train = torch::zeros({train_limit});

    #pragma omp parallel for
    for (int i = 0; i < train_limit; ++i) {
        for (int j = 0; j < num_features; ++j) {
            input_train[j][i] = input_data_rows[i][j];
        }
        output_train[i] = output_data_vec[i];
    }

    torch::Tensor input_test = torch::zeros({num_features, num_test});
    torch::Tensor output_test = torch::zeros({num_test});

    #pragma omp parallel for
    for (int i = 0; i < num_test; ++i) {
        for (int j = 0; j < num_features; ++j) {
            input_test[j][i] = input_data_rows[train_limit + i][j];
        }
        output_test[i] = output_data_vec[train_limit + i];
    }

    std::cout << "Train samples: " << train_limit << std::endl;
    std::cout << "Test samples: " << num_test << std::endl;
    std::cout << "Input features: " << num_features << std::endl;

    input_train = input_train.to(device);
    output_train = output_train.to(device);
    input_test = input_test.to(device);
    output_test = output_test.to(device);

    // Normalize data
    auto normalizer = create_normalizer(norm_method);
    normalizer->fit(input_train, output_train);
    torch::Tensor inputn = normalizer->transform_X(input_train);
    torch::Tensor outputn = normalizer->transform_Y(output_train);

    std::cout << "\nUsing normalization: " << normalizer->name() << std::endl;
    std::cout << "Normalized input range: [" << inputn.min().item<double>()
              << ", " << inputn.max().item<double>() << "]" << std::endl;
    std::cout << "Normalized output range: [" << outputn.min().item<double>()
              << ", " << outputn.max().item<double>() << "]" << std::endl;

    // Check for restart / checkpoint
    int start_iteration = 0;
    int start_epoch = 0;
    double saved_best_r2 = -1e9;
    bool has_checkpoint = controller.checkpoint_exists(checkpoint_meta);

    std::string initial_cmd = controller.read_command();
    if (initial_cmd == "restart") {
        controller.acknowledge_command();
        controller.clear_checkpoint(checkpoint_meta, checkpoint_model);
        has_checkpoint = false;
        std::cout << "Restart command received: starting fresh training." << std::endl;
    } else if (has_checkpoint) {
        nlohmann::json meta;
        if (controller.load_checkpoint_meta(checkpoint_meta, meta)) {
            start_iteration = meta.value("iteration", 0);
            start_epoch = meta.value("epoch", 0);
            saved_best_r2 = meta.value("best_r2", -1e9);
            std::cout << "Resuming from checkpoint: iteration " << (start_iteration + 1)
                      << ", epoch " << (start_epoch + 1) << std::endl;
        }
    }

    // Open training log file
    {
        std::ofstream train_log_init(output_training_log_path);
        train_log_init << "iteration,epoch,loss,learning_rate,optimizer,r2,rmse,mae\n";
    }

    // Training loop with early stopping
    std::shared_ptr<FeedForwardNet> best_net = nullptr;
    double best_r2 = saved_best_r2;

    controller.update_status("ffn", "running", 0, epochs, 0.0, best_r2, 0.0, 0.0,
                             "Training started");

    for (int pp = start_iteration; pp < max_iterations; ++pp) {
        std::cout << "\n=== Training Iteration " << (pp + 1) << " ===" << std::endl;

        auto net = std::make_shared<FeedForwardNet>(num_features, hidden_neurons, 1);
        net->to(device);

        // Load checkpoint model if resuming
        if (pp == start_iteration && has_checkpoint) {
            try {
                torch::load(net, checkpoint_model);
                std::cout << "Loaded checkpoint model." << std::endl;
            } catch (...) {
                std::cerr << "Warning: failed to load checkpoint model. Starting from scratch." << std::endl;
            }
        }

        torch::Tensor X_train = inputn.transpose(0, 1);
        torch::Tensor Y_train = outputn.unsqueeze(1);

        std::cout << "Using optimizer: " << optimizer_type << std::endl;
        net->train();

        std::ofstream train_log(output_training_log_path, std::ios::app);

        // Optimizer setup
        std::unique_ptr<torch::optim::Optimizer> optimizer;
        double learning_rate = 0.001;

        if (optimizer_type == "lbfgs") {
            auto lbfgs_config = config["optimizer"]["lbfgs"];
            learning_rate = lbfgs_config.value("learning_rate", 1.0);
            optimizer = std::make_unique<torch::optim::LBFGS>(
                net->parameters(),
                torch::optim::LBFGSOptions(learning_rate)
                    .max_iter(lbfgs_config.value("max_iter", 20))
                    .max_eval(lbfgs_config.value("max_eval", 25))
                    .tolerance_grad(lbfgs_config.value("tolerance_grad", 1e-7))
                    .tolerance_change(lbfgs_config.value("tolerance_change", 1e-9))
                    .history_size(lbfgs_config.value("history_size", 100))
            );
        } else if (optimizer_type == "rmsprop") {
            auto rmsprop_config = config["optimizer"]["rmsprop"];
            learning_rate = rmsprop_config.value("learning_rate", 0.01);
            optimizer = std::make_unique<torch::optim::RMSprop>(
                net->parameters(),
                torch::optim::RMSpropOptions(learning_rate)
                    .alpha(rmsprop_config.value("alpha", 0.99))
                    .eps(rmsprop_config.value("eps", 1e-8))
                    .weight_decay(rmsprop_config.value("weight_decay", 0.0))
                    .momentum(rmsprop_config.value("momentum", 0.0))
                    .centered(rmsprop_config.value("centered", false))
            );
        } else if (optimizer_type == "adam") {
            auto adam_config = config["optimizer"]["adam"];
            learning_rate = adam_config.value("learning_rate", 0.001);
            optimizer = std::make_unique<torch::optim::Adam>(
                net->parameters(),
                torch::optim::AdamOptions(learning_rate)
                    .betas({adam_config.value("beta1", 0.9), adam_config.value("beta2", 0.999)})
                    .eps(adam_config.value("eps", 1e-8))
                    .weight_decay(adam_config.value("weight_decay", 0.0))
            );
        } else if (optimizer_type == "adamw") {
            auto adamw_config = config["optimizer"]["adamw"];
            learning_rate = adamw_config.value("learning_rate", 0.001);
            optimizer = std::make_unique<torch::optim::AdamW>(
                net->parameters(),
                torch::optim::AdamWOptions(learning_rate)
                    .betas(std::make_tuple(adamw_config.value("beta1", 0.9), adamw_config.value("beta2", 0.999)))
                    .eps(adamw_config.value("eps", 1e-8))
                    .weight_decay(adamw_config.value("weight_decay", 0.001))
            );
        } else {
            std::cerr << "Unknown optimizer type: " << optimizer_type << std::endl;
            controller.update_status("ffn", "stopped", 0, epochs, 0.0, best_r2, 0.0, 0.0,
                                     "Unknown optimizer");
            return;
        }

        for (int epoch = start_epoch; epoch < epochs; ++epoch) {
            // Check control command
            std::string cmd = controller.read_command();
            if (cmd == "pause") {
                controller.acknowledge_command();
                controller.update_status("ffn", "paused", epoch, epochs, 0.0, best_r2, 0.0, 0.0,
                                         "Paused by user");
                // Save checkpoint
                torch::save(net, checkpoint_model);
                nlohmann::json meta = {
                    {"iteration", pp},
                    {"epoch", epoch},
                    {"best_r2", best_r2},
                    {"hidden_layer_neurons", hidden_neurons}
                };
                controller.save_checkpoint_meta(checkpoint_meta, meta);
                std::cout << "Checkpoint saved. Waiting for resume..." << std::endl;

                std::string resume_cmd = controller.wait_for_resume();
                if (resume_cmd == "stop") {
                    std::cout << "Stop command received. Exiting." << std::endl;
                    controller.update_status("ffn", "stopped", epoch, epochs, 0.0, best_r2, 0.0, 0.0,
                                             "Stopped by user");
                    return;
                }
                if (resume_cmd == "restart") {
                    std::cout << "Restart command received. Clearing checkpoint and restarting." << std::endl;
                    controller.clear_checkpoint(checkpoint_meta, checkpoint_model);
                    controller.update_status("ffn", "running", 0, epochs, 0.0, 0.0, 0.0, 0.0,
                                             "Restarting fresh");
                    // Reset state
                    start_iteration = 0;
                    start_epoch = 0;
                    best_r2 = -1e9;
                    best_net = nullptr;
                    pp = -1;  // Will be incremented to 0
                    break;    // Break inner epoch loop
                }
                controller.update_status("ffn", "running", epoch, epochs, 0.0, best_r2, 0.0, 0.0,
                                         "Resumed");
            }
            if (cmd == "stop") {
                controller.acknowledge_command();
                torch::save(net, checkpoint_model);
                nlohmann::json meta = {{"iteration", pp}, {"epoch", epoch}, {"best_r2", best_r2},
                                       {"hidden_layer_neurons", hidden_neurons}};
                controller.save_checkpoint_meta(checkpoint_meta, meta);
                std::cout << "Stop command received. Exiting." << std::endl;
                controller.update_status("ffn", "stopped", epoch, epochs, 0.0, best_r2, 0.0, 0.0,
                                         "Stopped by user");
                return;
            }
            if (cmd == "restart") {
                controller.acknowledge_command();
                controller.clear_checkpoint(checkpoint_meta, checkpoint_model);
                controller.update_status("ffn", "running", 0, epochs, 0.0, 0.0, 0.0, 0.0,
                                         "Restarting fresh");
                start_iteration = 0;
                start_epoch = 0;
                best_r2 = -1e9;
                best_net = nullptr;
                pp = -1;
                break;
            }

            // Training step
            torch::Tensor loss;
            if (optimizer_type == "lbfgs") {
                auto closure = [&]() -> torch::Tensor {
                    optimizer->zero_grad();
                    torch::Tensor output = net->forward(X_train);
                    torch::Tensor l = torch::mse_loss(output, Y_train);
                    l.backward();
                    torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
                    return l;
                };
                loss = static_cast<torch::optim::LBFGS*>(optimizer.get())->step(closure);
            } else {
                optimizer->zero_grad();
                torch::Tensor output = net->forward(X_train);
                loss = torch::mse_loss(output, Y_train);
                loss.backward();
                torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
                optimizer->step();
            }

            // Logging
            train_log << (pp + 1) << "," << (epoch + 1) << ","
                     << loss.item<double>() << "," << learning_rate << ","
                     << optimizer_type << ",0,0,0\n";

            if ((epoch + 1) % print_interval == 0) {
                std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss.item<double>() << std::endl;
            }

            if (loss.item<double>() < goal_loss) {
                std::cout << "Reached goal at epoch " << (epoch + 1) << std::endl;
                break;
            }

            // Update status periodically
            if ((epoch + 1) % print_interval == 0) {
                controller.update_status("ffn", "running", epoch + 1, epochs,
                                         loss.item<double>(), best_r2, 0.0, 0.0,
                                         "Iteration " + std::to_string(pp + 1));
            }
        }

        train_log.close();

        // If we broke due to restart, skip validation and restart outer loop
        if (pp < 0) {
            pp = -1;
            continue;
        }

        // Reset start_epoch for next iteration
        start_epoch = 0;

        // Validation
        std::cout << "\n--- Validation ---" << std::endl;
        net->eval();
        torch::NoGradGuard no_grad;

        int kk = num_test;
        std::vector<double> data_pre;
        torch::Tensor input = input_test.index({torch::indexing::Slice(), 0}).clone();

        for (int n = 0; n < kk; ++n) {
            torch::Tensor inputn_new = normalizer->transform_X(input);
            torch::Tensor an = net->forward(inputn_new.unsqueeze(0));
            torch::Tensor BPoutput = normalizer->inverse_transform_Y(an);
            double prediction = BPoutput[0][0].item<double>();
            data_pre.push_back(prediction);

            if (n < kk - 1) {
                input = input_test.index({torch::indexing::Slice(), n + 1}).clone();
                int dataset_row_size = static_cast<int>(input_columns.size()) + 1;
                int update_idx = dataset_row_size * w - dataset_row_size - 1;
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
        double MAE = calculateMAE(output_test, YPred);

        std::cout << "RMSE: " << RMSE << std::endl;
        std::cout << "MAE:  " << MAE << std::endl;
        std::cout << "R-squared value : " << R2 << std::endl;

        controller.update_status("ffn", "running", epochs, epochs, 0.0, best_r2, RMSE, MAE,
                                 "Validation R2=" + std::to_string(R2));

        if (R2 > best_r2) {
            best_r2 = R2;
            best_net = net;
        }

        if (R2 > target_r2) {
            std::cout << "\nTarget R2 reached! Saving model..." << std::endl;
            torch::save(net, output_model_path);
            controller.clear_checkpoint(checkpoint_meta, checkpoint_model);
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
            torch::Tensor inputn_new = normalizer->transform_X(input);
            torch::Tensor an = best_net->forward(inputn_new.unsqueeze(0));
            torch::Tensor BPoutput = normalizer->inverse_transform_Y(an);
            double prediction = BPoutput[0][0].item<double>();
            data_pre.push_back(prediction);

            if (n < kk - 1) {
                input = input_test.index({torch::indexing::Slice(), n + 1}).clone();
                int dataset_row_size = static_cast<int>(input_columns.size()) + 1;
                int update_idx = dataset_row_size * w - dataset_row_size - 1;
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
        double MAE_val = calculateMAE(output_test, YPred);

        std::cout << "Final RMSE: " << RMSE << std::endl;
        std::cout << "Final MAE:  " << MAE_val << std::endl;
        std::cout << "Final R2:   " << R2 << std::endl;

        std::ofstream outfile(output_predictions_path);
        outfile << "YTest,YPred,Error\n";
        for (int i = 0; i < kk; ++i) {
            double y_test = output_test[i].item<double>();
            double y_pred = YPred[i].item<double>();
            double error = y_test - y_pred;
            outfile << y_test << "," << y_pred << "," << error << "\n";
        }
        outfile.close();
        std::cout << "\nPredictions saved to " << output_predictions_path << std::endl;

        controller.update_status("ffn", "completed", epochs, epochs, 0.0, R2, RMSE, MAE_val,
                                 "Training completed");
    }

    std::cout << "\nFFN training and validation completed!" << std::endl;
}

} // namespace tju_torch

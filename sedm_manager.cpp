#include "sedm_manager.h"
#include "common_ffn.h"
#include "training_controller.h"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <omp.h>

namespace tju_torch {

sedmInputParameter parse_sedm_input_parameter(const nlohmann::json& config) {
    sedmInputParameter p;

    p.nn = config.value("nn", p.nn);
    p.A_cell = config.value("A_cell", p.A_cell);
    p.t_MEM = config.value("t_MEM", p.t_MEM);
    p.t_CLc = config.value("t_CLc", p.t_CLc);
    p.t_MPLc = config.value("t_MPLc", p.t_MPLc);
    p.t_GDLc = config.value("t_GDLc", p.t_GDLc);
    p.t_CHc = config.value("t_CHc", p.t_CHc);
    p.POR_CLc = config.value("POR_CLc", p.POR_CLc);
    p.POR_MPLc = config.value("POR_MPLc", p.POR_MPLc);
    p.POR_GDLc = config.value("POR_GDLc", p.POR_GDLc);
    p.Alpha_a = config.value("Alpha_a", p.Alpha_a);
    p.Alpha_c = config.value("Alpha_c", p.Alpha_c);
    p.j_ref_a = config.value("j_ref_a", p.j_ref_a);
    p.j_ref_c = config.value("j_ref_c", p.j_ref_c);
    p.K_c_ini = config.value("K_c_ini", p.K_c_ini);
    p.b_leak = config.value("b_leak", p.b_leak);
    p.b_ECSA = config.value("b_ECSA", p.b_ECSA);
    p.b_ion = config.value("b_ion", p.b_ion);
    p.b_R = config.value("b_R", p.b_R);
    p.b_D = config.value("b_D", p.b_D);
    p.b_B = config.value("b_B", p.b_B);

    return p;
}

static nlohmann::json get_lbfgs_config(const nlohmann::json& config) {
    if (!config.contains("optimizer")) {
        return nlohmann::json::object();
    }
    const auto& optimizer = config.at("optimizer");
    if (!optimizer.is_object() || !optimizer.contains("lbfgs")) {
        return nlohmann::json::object();
    }
    const auto& lbfgs = optimizer.at("lbfgs");
    if (!lbfgs.is_object()) {
        return nlohmann::json::object();
    }
    return lbfgs;
}

static nlohmann::json get_adamw_config(const nlohmann::json& config) {
    if (!config.contains("optimizer")) {
        return nlohmann::json::object();
    }
    const auto& optimizer = config.at("optimizer");
    if (!optimizer.is_object() || !optimizer.contains("adamw")) {
        return nlohmann::json::object();
    }
    const auto& adamw = optimizer.at("adamw");
    if (!adamw.is_object()) {
        return nlohmann::json::object();
    }
    return adamw;
}

// SEDM (Semi-Empirical Dynamic Model) function
// Parameter declaration order matches Prediction_model_2.m; values from sedmInputParameter where configurable.
static double SEDM(const sedmInputParameter& p, double tt, double Pc, double Pa, double T, double I) {
    const int nn = p.nn;
    const double A_cell = p.A_cell;
    const double L_Pt = 4;
    const double F = 96487;
    const double R = 8.314472;
    const double P0 = 101325;
    const double Alpha_c = p.Alpha_c;
    const double Alpha_a = p.Alpha_a;
    const double Gamma_a = 0.5;
    const double Gamma_c = 1.0;
    const double c_o2_ref = 3.39;
    const double t_MEM = p.t_MEM;
    const double t_CLc = p.t_CLc;
    const double t_MPLc = p.t_MPLc;
    const double t_GDLc = p.t_GDLc;
    const double t_CHc = p.t_CHc;
    const double POR_CLc = p.POR_CLc;
    const double POR_MPLc = p.POR_MPLc;
    const double POR_GDLc = p.POR_GDLc;
    const double j_ref_a = p.j_ref_a;
    const double j_ref_c = p.j_ref_c;
    const double b_leak = p.b_leak;
    const double b_ECSA = p.b_ECSA;
    const double b_ion = p.b_ion;
    const double b_R = p.b_R;
    const double b_D = p.b_D;
    const double b_B = p.b_B;

    double r_leak = std::exp(b_leak * tt);
    double r_ECSA = std::exp(b_ECSA * tt);
    double r_ion = std::exp(b_ion * tt);

    double i_leak_ini = 20.0 * A_cell;
    double A_ECSA_ini = 60 * (A_cell * L_Pt);
    double R_ion_ini = 100e-7 / A_cell;
    double R_ele_ini = 20e-7 / A_cell;
    double D_o2_ini = 2.652e-5 * std::pow(T / 333.15, 1.5) * (1.0 / Pc) * std::pow(POR_GDLc, 1.5);
    double K_c_ini = p.K_c_ini;

    double i_leak = i_leak_ini * r_leak;
    double A_ECSA = A_ECSA_ini * r_ECSA;
    double R_total = R_ion_ini * r_ion + (R_ele_ini + b_R * tt);
    double D_o2 = D_o2_ini + b_D * tt;
    double K_c = K_c_ini + b_B * tt;

    double E_nernst = 1.229 - 0.846e-3 * (T - 298.15) +
                      R * T / 2.0 / F * (std::log(Pa) + 0.5 * std::log(Pc * 0.21));

    double b_a = R * T / (2.0 * Alpha_a * F);
    double theta_T_a = std::exp(-1400.0 * (1.0 / T - 1.0 / 298.15));
    double c_h2_CLa = Pa * P0 / R / T;
    double k_ele_a = j_ref_a * std::pow(c_h2_CLa / c_o2_ref, Gamma_a) * theta_T_a;
    double V_act_a = b_a * (i_leak + I) / A_ECSA / k_ele_a;

    double b_c = R * T / (4.0 * Alpha_c * F);
    double theta_T_c = std::exp(-7900.0 * (1.0 / T - 1.0 / 298.15));
    double c_o2_CLc = 0.21 * Pc * P0 / R / T;
    double k_ele_c = j_ref_c * std::pow(c_o2_CLc / c_o2_ref, Gamma_c) * theta_T_c;
    double V_act_c = -b_c * std::log((i_leak + I) / A_ECSA / k_ele_c);

    double V_ohm = -I * R_total;

    double D_o2_GDLc = 2.652e-5 * std::pow(T / 333.15, 1.5) * (1.0 / Pc) * std::pow(POR_GDLc, 1.5);
    double P_o2 = Pc * 0.21 * P0;
    double I_lim = 4.0 * F * (D_o2_GDLc / t_GDLc) * (P_o2 / R / T);
    double term_c = 1.0 - (I / A_ECSA) / I_lim;
    double V_conc_c = K_c * b_c * std::log(term_c);

    double V_cell_sim = E_nernst + V_act_a + V_act_c + V_ohm + V_conc_c;
    double V_stack_sim = V_cell_sim * static_cast<double>(nn);

    return V_stack_sim;
}

    void BatteryLifespanManager::run(const nlohmann::json &config) {
        std::string submode = config.value("submode", "predict");
        if (submode == "train") {
            runTrain(config);
        } else if (submode == "predict") {
            runPredict(config);
        } else {
            std::cerr << "Error: Unknown submode '" << submode << "'" << std::endl;
            std::cerr << "Valid submodes: 'train', 'predict'" << std::endl;
        }
    }

    // ============================================================================
    // runTrain — iterative DDM training with AdamW or LBFGS
    // ============================================================================
    void BatteryLifespanManager::runTrain(const nlohmann::json &config) {
        std::cout << std::fixed << std::setprecision(6);

        // All file paths must be explicitly configured in JSON
        if (!config.contains("input_data_path")) {
            std::cerr << "Error: config missing 'input_data_path'" << std::endl;
            return;
        }
        if (!config.contains("model_path")) {
            std::cerr << "Error: config missing 'model_path'" << std::endl;
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
        std::string model_path = config["model_path"];
        std::string output_predictions_path = config["output_predictions_path"];
        std::string output_training_log_path = config["output_training_log_path"];
        std::string control_file = config["control_file_path"];
        std::string status_file = config["status_file_path"];

        int num_rows_begin = 0;
        int num_rows_end = -1;
        if (config.contains("num_rows_begin") || config.contains("num_rows_end")) {
            num_rows_begin = config.value("num_rows_begin", 0);
            num_rows_end = config.value("num_rows_end", -1);
        } else if (config.contains("num_rows")) {
            num_rows_end = config.value("num_rows", 900);
        }
        int window_size = config.value("window_size", 5);
        int epochs = config.value("epochs", 1000);
        double goal_loss = config.value("goal_loss", 1e-10);
        int max_iterations = config.value("max_iterations", 1000);
        double target_r2 = config.value("target_r2", 0.85);
        int print_interval = config.value("print_interval", 100);
        int batch_size = config.value("batch_size", 32);
        if (batch_size <= 0) batch_size = 32;

        std::string optimizer_type = config.value("optimizer_type", "lbfgs");

        // Column configuration
        std::vector<int> input_columns;
        if (config.contains("input_columns") && config["input_columns"].is_array()) {
            for (const auto &item: config["input_columns"]) {
                input_columns.push_back(item.get<int>());
            }
        }
        if (input_columns.empty()) {
            input_columns = {4, 5, 8, 10};
        }
        int output_column = config.value("output_column", 11);

        int max_col = output_column;
        for (int col: input_columns) {
            if (col > max_col) max_col = col;
        }
        int min_cols = max_col + 1;

        // hidden_layer_neurons
        std::vector<int64_t> hidden_neurons;
        if (config.contains("hidden_layer_neurons") && config["hidden_layer_neurons"].is_array()) {
            for (const auto &item: config["hidden_layer_neurons"]) {
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
        std::string checkpoint_model = model_path + ".checkpoint.pt";
        std::string checkpoint_meta = model_path + ".checkpoint.json";

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
        auto raw_data = readDataFile(data_file, num_rows_begin, num_rows_end);
        if (raw_data.empty()) {
            std::cerr << "Error: No data loaded!" << std::endl;
            controller.update_status("battery_lifespan", "stopped", 0, epochs, 0.0, 0.0, 0.0, 0.0, "Data load failed");
            return;
        }
        std::cout << "Loaded " << raw_data.size() << " rows of data." << std::endl;

        const int w = window_size;
        const size_t dd = raw_data.size();

        // Prepare Input and Output for neural network
        std::vector<std::vector<double> > Input, Output;
        for (const auto &row: raw_data) {
            if (row.size() >= static_cast<size_t>(min_cols)) {
                std::vector<double> in_row;
                for (int col: input_columns) {
                    in_row.push_back(row[col]);
                }
                Input.push_back(in_row);
                Output.push_back({row[output_column]});
            }
        }

        std::vector<std::vector<double> > Dataset;
        for (size_t i = 0; i < Input.size(); ++i) {
            std::vector<double> row = Input[i];
            row.push_back(Output[i][0]);
            Dataset.push_back(row);
        }

        // Apply sliding window
        std::vector<std::vector<double> > input_data_rows;
        std::vector<double> output_data_vec;

        for (size_t i = 0; i < dd - w; ++i) {
            std::vector<double> Input_pre;
            for (int j = 0; j < w; ++j) {
                size_t idx = i + j;
                if (idx < Dataset.size()) {
                    for (double val: Dataset[idx]) {
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

        int num_samples = static_cast<int>(input_data_rows.size());
        int num_features = static_cast<int>(input_data_rows[0].size());

        int numTimeStepsTrain = 300;
        if (config.contains("training_sample_ratio")) {
            double ratio = config.value("training_sample_ratio", 0.5);
            numTimeStepsTrain = static_cast<int>(std::round(ratio * num_samples));
        } else if (config.contains("train_samples")) {
            numTimeStepsTrain = config.value("train_samples", 300);
        }
        if (numTimeStepsTrain <= 0) numTimeStepsTrain = 1;
        if (numTimeStepsTrain > num_samples) numTimeStepsTrain = num_samples;

        int num_test = num_samples - numTimeStepsTrain;

        torch::Tensor input_train = torch::zeros({num_features, numTimeStepsTrain});
        torch::Tensor output_train = torch::zeros({numTimeStepsTrain});

        int train_limit = std::min(numTimeStepsTrain, num_samples);
        for (int i = 0; i < train_limit; ++i) {
            for (int j = 0; j < num_features; ++j) {
                input_train[j][i] = input_data_rows[i][j];
            }
            output_train[i] = output_data_vec[i];
        }

        torch::Tensor input_test = torch::zeros({num_features, num_test});
        torch::Tensor output_test = torch::zeros({num_test});

        for (int i = 0; i < num_test; ++i) {
            for (int j = 0; j < num_features; ++j) {
                input_test[j][i] = input_data_rows[train_limit + i][j];
            }
            output_test[i] = output_data_vec[train_limit + i];
        }

        input_train = input_train.to(device);
        output_train = output_train.to(device);
        input_test = input_test.to(device);
        output_test = output_test.to(device);

        std::cout << "Train samples: " << train_limit << std::endl;
        std::cout << "Test samples: " << num_test << std::endl;

        // Normalize data
        auto normalizer = create_normalizer(norm_method);
        normalizer->fit(input_train, output_train);
        torch::Tensor inputn = normalizer->transform_X(input_train);
        torch::Tensor outputn = normalizer->transform_Y(output_train);

        std::cout << "Using normalization: " << normalizer->name() << std::endl;

        // Check for restart / checkpoint
        int start_iteration = 0;
        int start_epoch = 0;
        double saved_best_r2 = -1e9;
        int saved_seed = 0;
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
                saved_seed = meta.value("seed", 0);
                std::cout << "Resuming from checkpoint: iteration " << (start_iteration + 1)
                        << ", epoch " << (start_epoch + 1);
                if (saved_seed > 0) {
                    std::cout << ", seed " << saved_seed;
                }
                std::cout << std::endl;
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

        controller.update_status("battery_lifespan", "running", 0, epochs, 0.0, best_r2, 0.0, 0.0,
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
            int num_train_samples = static_cast<int>(X_train.size(0));

            net->train();

            std::ofstream train_log(output_training_log_path, std::ios::app);

            bool training_success = false;
            bool user_stopped = false;
            double last_loss = std::numeric_limits<double>::quiet_NaN();

            // =====================================================================
            // Optimizer-specific training
            // =====================================================================
            if (optimizer_type == "lbfgs") {
                const nlohmann::json lbfgs_config = get_lbfgs_config(config);
                double lr = lbfgs_config.value("learning_rate", 1.0);

                if (batch_size < num_train_samples) {
                    std::cout << "Note: LBFGS uses full-batch training. Configured batch_size ("
                              << batch_size << ") is ignored for LBFGS." << std::endl;
                }

                int base_seed = 42 + pp * 10;
                int seed_start = base_seed;
                if (has_checkpoint && pp == start_iteration && saved_seed > 0) {
                    seed_start = saved_seed;
                }

                for (int seed = seed_start; seed < base_seed + 10 && !training_success && !user_stopped; ++seed) {
                    if (seed > seed_start) {
                        std::cout << "Training diverged with seed " << (seed - 1)
                                << ", retrying with seed " << seed << "..." << std::endl;
                    }

                    torch::manual_seed(seed);
                    net = std::make_shared<FeedForwardNet>(num_features, hidden_neurons, 1);
                    net->to(device);
                    net->train();

                    torch::optim::LBFGS optimizer(
                        net->parameters(),
                        torch::optim::LBFGSOptions(lr)
                        .max_iter(lbfgs_config.value("max_iter", 20))
                        .max_eval(lbfgs_config.value("max_eval", 25))
                        .tolerance_grad(lbfgs_config.value("tolerance_grad", 1e-7))
                        .tolerance_change(lbfgs_config.value("tolerance_change", 1e-9))
                        .history_size(lbfgs_config.value("history_size", 100))
                    );

                    last_loss = std::numeric_limits<double>::quiet_NaN();
                    int epoch_start = (seed == seed_start && pp == start_iteration && has_checkpoint) ? start_epoch : 0;

                    for (int epoch = epoch_start; epoch < epochs; ++epoch) {
                        // Check control command
                        std::string cmd = controller.read_command();
                        if (cmd == "pause") {
                            controller.acknowledge_command();
                            controller.update_status("battery_lifespan", "paused", epoch, epochs, 0.0, best_r2, 0.0,
                                                     0.0,
                                                     "Paused by user");
                            torch::save(net, checkpoint_model);
                            nlohmann::json meta = {
                                {"iteration", pp},
                                {"epoch", epoch},
                                {"best_r2", best_r2},
                                {"seed", seed}
                            };
                            controller.save_checkpoint_meta(checkpoint_meta, meta);
                            std::cout << "Checkpoint saved. Waiting for resume..." << std::endl;

                            std::string resume_cmd = controller.wait_for_resume();
                            if (resume_cmd == "stop") {
                                std::cout << "Stop command received. Exiting." << std::endl;
                                controller.update_status("battery_lifespan", "stopped", epoch, epochs, 0.0, best_r2,
                                                         0.0, 0.0,
                                                         "Stopped by user");
                                return;
                            }
                            if (resume_cmd == "restart") {
                                std::cout << "Restart command received. Clearing checkpoint and restarting." <<
                                        std::endl;
                                controller.clear_checkpoint(checkpoint_meta, checkpoint_model);
                                controller.update_status("battery_lifespan", "running", 0, epochs, 0.0, 0.0, 0.0, 0.0,
                                                         "Restarting fresh");
                                start_iteration = 0;
                                start_epoch = 0;
                                best_r2 = -1e9;
                                best_net = nullptr;
                                pp = -1;
                                break;
                            }
                            controller.update_status("battery_lifespan", "running", epoch, epochs, 0.0, best_r2, 0.0,
                                                     0.0,
                                                     "Resumed");
                        }
                        if (cmd == "stop") {
                            controller.acknowledge_command();
                            torch::save(net, checkpoint_model);
                            nlohmann::json meta = {
                                {"iteration", pp},
                                {"epoch", epoch},
                                {"best_r2", best_r2},
                                {"seed", seed}
                            };
                            controller.save_checkpoint_meta(checkpoint_meta, meta);
                            std::cout << "Stop command received. Exiting." << std::endl;
                            controller.update_status("battery_lifespan", "stopped", epoch, epochs, 0.0, best_r2, 0.0,
                                                     0.0,
                                                     "Stopped by user");
                            return;
                        }
                        if (cmd == "restart") {
                            controller.acknowledge_command();
                            controller.clear_checkpoint(checkpoint_meta, checkpoint_model);
                            controller.update_status("battery_lifespan", "running", 0, epochs, 0.0, 0.0, 0.0, 0.0,
                                                     "Restarting fresh");
                            start_iteration = 0;
                            start_epoch = 0;
                            best_r2 = -1e9;
                            best_net = nullptr;
                            pp = -1;
                            break;
                        }

                        auto closure = [&]() -> torch::Tensor {
                            optimizer.zero_grad();
                            torch::Tensor output = net->forward(X_train);
                            torch::Tensor loss = torch::mse_loss(output, Y_train);
                            loss.backward();
                            //torch::nn::utils::clip_grad_norm_(net->parameters(), 10.0);
                            return loss;
                        };

                        torch::Tensor loss = optimizer.step(closure);
                        last_loss = loss.item<double>();

                        if (std::isnan(last_loss) || std::isinf(last_loss)) {
                            std::cout << "Diverged at epoch " << (epoch + 1) << " (loss=" << last_loss <<
                                    "), aborting retry..." << std::endl;
                            break;
                        }

                        train_log << (pp + 1) << "," << (epoch + 1) << ","
                                << last_loss << "," << lr << ",lbfgs,0,0,0\n";

                        if ((epoch + 1) % print_interval == 0) {
                            std::cout << "Epoch " << (epoch + 1) << ", Loss: " << last_loss << std::endl;
                        }
                        if (last_loss < goal_loss) {
                            std::cout << "Converged at epoch " << (epoch + 1) << std::endl;
                            break;
                        }

                        if ((epoch + 1) % print_interval == 0) {
                            controller.update_status("battery_lifespan", "running", epoch + 1, epochs,
                                                     last_loss, best_r2, 0.0, 0.0,
                                                     "Iteration " + std::to_string(pp + 1) + ", seed " + std::to_string(
                                                         seed));
                        }
                    }

                    // If we broke due to restart, skip validation
                    if (pp < 0) {
                        break;
                    }

                    if (!user_stopped && !std::isnan(last_loss) && !std::isinf(last_loss)) {
                        training_success = true;
                    }
                }
            } else if (optimizer_type == "adamw") {
                const nlohmann::json adamw_config = get_adamw_config(config);
                double lr = adamw_config.value("learning_rate", 0.001);
                torch::optim::AdamW optimizer(
                    net->parameters(),
                    torch::optim::AdamWOptions(lr)
                    .betas(std::make_tuple(adamw_config.value("beta1", 0.9), adamw_config.value("beta2", 0.999)))
                    .eps(adamw_config.value("eps", 1e-8))
                    .weight_decay(adamw_config.value("weight_decay", 0.001))
                );

                int epoch_start = (pp == start_iteration && has_checkpoint) ? start_epoch : 0;

                for (int epoch = epoch_start; epoch < epochs; ++epoch) {
                    // Check control command
                    std::string cmd = controller.read_command();
                    if (cmd == "pause") {
                        controller.acknowledge_command();
                        controller.update_status("battery_lifespan", "paused", epoch, epochs, 0.0, best_r2, 0.0, 0.0,
                                                 "Paused by user");
                        torch::save(net, checkpoint_model);
                        nlohmann::json meta = {
                            {"iteration", pp},
                            {"epoch", epoch},
                            {"best_r2", best_r2},
                            {"seed", 0}
                        };
                        controller.save_checkpoint_meta(checkpoint_meta, meta);
                        std::cout << "Checkpoint saved. Waiting for resume..." << std::endl;

                        std::string resume_cmd = controller.wait_for_resume();
                        if (resume_cmd == "stop") {
                            std::cout << "Stop command received. Exiting." << std::endl;
                            controller.update_status("battery_lifespan", "stopped", epoch, epochs, 0.0, best_r2, 0.0,
                                                     0.0,
                                                     "Stopped by user");
                            return;
                        }
                        if (resume_cmd == "restart") {
                            std::cout << "Restart command received. Clearing checkpoint and restarting." << std::endl;
                            controller.clear_checkpoint(checkpoint_meta, checkpoint_model);
                            controller.update_status("battery_lifespan", "running", 0, epochs, 0.0, 0.0, 0.0, 0.0,
                                                     "Restarting fresh");
                            start_iteration = 0;
                            start_epoch = 0;
                            best_r2 = -1e9;
                            best_net = nullptr;
                            pp = -1;
                            break;
                        }
                        controller.update_status("battery_lifespan", "running", epoch, epochs, 0.0, best_r2, 0.0, 0.0,
                                                 "Resumed");
                    }
                    if (cmd == "stop") {
                        controller.acknowledge_command();
                        torch::save(net, checkpoint_model);
                        nlohmann::json meta = {
                            {"iteration", pp},
                            {"epoch", epoch},
                            {"best_r2", best_r2},
                            {"seed", 0}
                        };
                        controller.save_checkpoint_meta(checkpoint_meta, meta);
                        std::cout << "Stop command received. Exiting." << std::endl;
                        controller.update_status("battery_lifespan", "stopped", epoch, epochs, 0.0, best_r2, 0.0, 0.0,
                                                 "Stopped by user");
                        return;
                    }
                    if (cmd == "restart") {
                        controller.acknowledge_command();
                        controller.clear_checkpoint(checkpoint_meta, checkpoint_model);
                        controller.update_status("battery_lifespan", "running", 0, epochs, 0.0, 0.0, 0.0, 0.0,
                                                 "Restarting fresh");
                        start_iteration = 0;
                        start_epoch = 0;
                        best_r2 = -1e9;
                        best_net = nullptr;
                        pp = -1;
                        break;
                    }

                    double epoch_loss = 0.0;
                    int effective_batch_size = std::min(batch_size, num_train_samples);
                    int num_batches = (num_train_samples + effective_batch_size - 1) / effective_batch_size;

                    for (int b = 0; b < num_batches; ++b) {
                        int start = b * effective_batch_size;
                        int end = std::min(start + effective_batch_size, num_train_samples);
                        auto X_batch = X_train.slice(0, start, end);
                        auto Y_batch = Y_train.slice(0, start, end);

                        optimizer.zero_grad();
                        torch::Tensor output = net->forward(X_batch);
                        torch::Tensor loss = torch::mse_loss(output, Y_batch);
                        loss.backward();
                        torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
                        optimizer.step();

                        epoch_loss += loss.item<double>() * (end - start);
                    }

                    double loss_val = epoch_loss / num_train_samples;

                    train_log << (pp + 1) << "," << (epoch + 1) << ","
                            << loss_val << "," << lr << ",adamw,0,0,0\n";

                    if ((epoch + 1) % print_interval == 0) {
                        std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss_val << std::endl;
                    }
                    if (loss_val < goal_loss) {
                        std::cout << "Reached goal at epoch " << (epoch + 1) << std::endl;
                        break;
                    }

                    if ((epoch + 1) % print_interval == 0) {
                        controller.update_status("battery_lifespan", "running", epoch + 1, epochs,
                                                 loss_val, best_r2, 0.0, 0.0,
                                                 "Training iteration " + std::to_string(pp + 1));
                    }
                }

                // AdamW does not use seed-retry; training is considered successful
                // unless we broke due to restart
                if (pp >= 0) {
                    training_success = true;
                }
            } else {
                std::cerr << "Unknown optimizer type: " << optimizer_type << std::endl;
                controller.update_status("battery_lifespan", "stopped", 0, epochs, 0.0, best_r2, 0.0, 0.0,
                                         "Unknown optimizer");
                return;
            }

            train_log.close();

            // If we broke due to restart, skip validation and restart outer loop
            if (pp < 0) {
                pp = -1;
                continue;
            }

            if (user_stopped) {
                return;
            }

            if (!training_success) {
                std::cerr << "Error: Training failed after multiple retries. All seeds produced NaN/Inf." << std::endl;
                controller.update_status("battery_lifespan", "stopped", 0, epochs, 0.0, 0.0, 0.0, 0.0,
                                         "Training failed - NaN");
                return;
            }

            // Reset start_epoch for next iteration
            start_epoch = 0;

            // Validation (DDM only, autoregressive)
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

            controller.update_status("battery_lifespan", "running", epochs, epochs, 0.0, best_r2, RMSE, MAE,
                                     "Validation R2=" + std::to_string(R2));

            if (R2 > best_r2) {
                best_r2 = R2;
                best_net = net;
            }

            if (R2 > target_r2) {
                std::cout << "\nTarget R2 reached! Saving model..." << std::endl;
                torch::save(net, model_path);
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

            controller.update_status("battery_lifespan", "completed", epochs, epochs, 0.0, R2, RMSE, MAE_val,
                                     "Training completed");
        }

        std::cout << "\nBattery lifespan training completed!" << std::endl;
    }


    // ============================================================================
    // runPredict — hybrid prediction (DDM + SEDM) with EOL estimation
    // ============================================================================
    void BatteryLifespanManager::runPredict(const nlohmann::json &config) {
        std::cout << std::fixed << std::setprecision(6);

        // All file paths must be explicitly configured in JSON
        if (!config.contains("input_data_path")) {
            std::cerr << "Error: config missing 'input_data_path'" << std::endl;
            return;
        }
        if (!config.contains("model_path")) {
            std::cerr << "Error: config missing 'model_path'" << std::endl;
            return;
        }
        if (!config.contains("output_predictions_path")) {
            std::cerr << "Error: config missing 'output_predictions_path'" << std::endl;
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
        std::string model_path = config["model_path"];
        std::string output_predictions_path = config["output_predictions_path"];
        std::string control_file = config["control_file_path"];
        std::string status_file = config["status_file_path"];

        int num_rows_begin = 0;
        int num_rows_end = -1;
        if (config.contains("num_rows_begin") || config.contains("num_rows_end")) {
            num_rows_begin = config.value("num_rows_begin", 0);
            num_rows_end = config.value("num_rows_end", -1);
        } else if (config.contains("num_rows")) {
            num_rows_end = config.value("num_rows", 900);
        }
        int window_size = config.value("window_size", 5);
        double RR = config.value("rr", 4.0);
        double time_begin = config.value("time_begin", 0.0);
        double eol_threshold_ratio = config.value("eol_threshold_ratio", 0.80);

        sedmInputParameter input_params = parse_sedm_input_parameter(config);

        // Column configuration
        std::vector<int> input_columns;
        if (config.contains("input_columns") && config["input_columns"].is_array()) {
            for (const auto &item: config["input_columns"]) {
                input_columns.push_back(item.get<int>());
            }
        }
        if (input_columns.empty()) {
            input_columns = {4, 5, 8, 10};
        }
        int output_column = config.value("output_column", 11);
        int time_column = config.value("time_column", 0);

        int max_col = std::max(output_column, time_column);
        for (int col: input_columns) {
            if (col > max_col) max_col = col;
        }
        int min_cols = max_col + 1;

        std::vector<int64_t> hidden_neurons;
        if (config.contains("hidden_layer_neurons") && config["hidden_layer_neurons"].is_array()) {
            for (const auto &item: config["hidden_layer_neurons"]) {
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

        TrainingController controller(control_file, status_file);

        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            std::cout << "CUDA is available! Using GPU." << std::endl;
        } else {
            std::cout << "CUDA not available. Using CPU with OpenMP acceleration." << std::endl;
        }

        // Load data
        std::cout << "\nLoading data from " << data_file << "..." << std::endl;
        auto raw_data = readDataFile(data_file, num_rows_begin, num_rows_end);
        if (raw_data.empty()) {
            std::cerr << "Error: No data loaded!" << std::endl;
            controller.update_status("battery_lifespan", "stopped", 0, 0, 0.0, 0.0, 0.0, 0.0, "Data load failed");
            return;
        }
        std::cout << "Loaded " << raw_data.size() << " rows of data." << std::endl;

        const int w = window_size;
        const size_t dd = raw_data.size();

        // Extract raw experimental data for SEDM physics model
        std::vector<double> tt, Pa, Pc, T, I, V_cell_exp;
        for (const auto &row: raw_data) {
            if (row.size() >= static_cast<size_t>(min_cols) && input_columns.size() >= 4) {
                tt.push_back(row[time_column]);
                Pa.push_back(row[input_columns[1]]);
                Pc.push_back(row[input_columns[0]]);
                T.push_back(row[input_columns[2]] + 273.15);
                I.push_back(row[input_columns[3]]);
                V_cell_exp.push_back(row[output_column] / static_cast<double>(input_params.nn));
            }
        }

        // Prepare Input and Output for neural network
        std::vector<std::vector<double> > Input, Output;
        for (const auto &row: raw_data) {
            if (row.size() >= static_cast<size_t>(min_cols)) {
                std::vector<double> in_row;
                for (int col: input_columns) {
                    in_row.push_back(row[col]);
                }
                Input.push_back(in_row);
                Output.push_back({row[output_column]});
            }
        }

        std::vector<std::vector<double> > Dataset;
        for (size_t i = 0; i < Input.size(); ++i) {
            std::vector<double> row = Input[i];
            row.push_back(Output[i][0]);
            Dataset.push_back(row);
        }

        // Apply sliding window
        std::vector<std::vector<double> > input_data_rows;
        std::vector<double> output_data_vec;

        for (size_t i = 0; i < dd - w; ++i) {
            std::vector<double> Input_pre;
            for (int j = 0; j < w; ++j) {
                size_t idx = i + j;
                if (idx < Dataset.size()) {
                    for (double val: Dataset[idx]) {
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

        int num_samples = static_cast<int>(input_data_rows.size());
        int num_features = static_cast<int>(input_data_rows[0].size());

        int numTimeStepsTrain = 300;
        if (config.contains("training_sample_ratio")) {
            double ratio = config.value("training_sample_ratio", 0.5);
            numTimeStepsTrain = static_cast<int>(std::round(ratio * num_samples));
        } else if (config.contains("train_samples")) {
            numTimeStepsTrain = config.value("train_samples", 300);
        }
        if (numTimeStepsTrain <= 0) numTimeStepsTrain = 1;
        if (numTimeStepsTrain > num_samples) numTimeStepsTrain = num_samples;

        int num_test = num_samples - numTimeStepsTrain;

        torch::Tensor input_train = torch::zeros({num_features, numTimeStepsTrain});
        torch::Tensor output_train = torch::zeros({numTimeStepsTrain});

        int train_limit = std::min(numTimeStepsTrain, num_samples);
        for (int i = 0; i < train_limit; ++i) {
            for (int j = 0; j < num_features; ++j) {
                input_train[j][i] = input_data_rows[i][j];
            }
            output_train[i] = output_data_vec[i];
        }

        torch::Tensor input_test = torch::zeros({num_features, num_test});
        torch::Tensor output_test = torch::zeros({num_test});

        for (int i = 0; i < num_test; ++i) {
            for (int j = 0; j < num_features; ++j) {
                input_test[j][i] = input_data_rows[train_limit + i][j];
            }
            output_test[i] = output_data_vec[train_limit + i];
        }

        input_train = input_train.to(device);
        output_train = output_train.to(device);
        input_test = input_test.to(device);
        output_test = output_test.to(device);

        std::cout << "Train samples: " << numTimeStepsTrain << std::endl;
        std::cout << "Test samples: " << num_test << std::endl;

        // Normalize data
        auto normalizer = create_normalizer(norm_method);
        normalizer->fit(input_train, output_train);

        std::cout << "Using normalization: " << normalizer->name() << std::endl;

        // Load pre-trained model (must exist)
        std::cout << "\n=== Loading Pre-trained Model ===" << std::endl;
        auto net = std::make_shared<FeedForwardNet>(num_features, hidden_neurons, 1);
        net->to(device);

        bool model_loaded = false;
        try {
            torch::load(net, model_path);
            std::cout << "Loaded pre-trained model from " << model_path << std::endl;
            model_loaded = true;
        } catch (...) {
            std::cerr << "Error: Could not load pre-trained model from " << model_path << std::endl;
            std::cerr << "Please run train submode first." << std::endl;
            controller.update_status("battery_lifespan", "stopped", 0, 0, 0.0, 0.0, 0.0, 0.0, "Model load failed");
            return;
        }

        // Hybrid prediction (DDM + SEDM)
        std::cout << "\n=== Hybrid Prediction (DDM + SEDM) ===" << std::endl;
        net->eval();
        torch::NoGradGuard no_grad;

        std::vector<double> aV_DDM, aV_SEM, aV_hybrid;
        std::vector<double> YTest_vec;

        for (int i = 0; i < num_test; ++i) {
            YTest_vec.push_back(output_test[i].item<double>());
        }

        torch::Tensor input = input_test.index({torch::indexing::Slice(), 0}).clone();

        controller.update_status("battery_lifespan", "running", 0, 0, 0.0, 0.0, 0.0, 0.0,
                                 "Hybrid prediction in progress");

        for (int n = 0; n < num_test; ++n) {
            if ((n + 1) % 50 == 0) {
                std::cout << "Processing step " << (n + 1) << "/" << num_test << std::endl;
                std::string cmd = controller.read_command();
                if (cmd == "stop") {
                    controller.acknowledge_command();
                    controller.update_status("battery_lifespan", "stopped", 0, 0, 0.0, 0.0, 0.0, 0.0,
                                             "Stopped during prediction");
                    return;
                }
            }

            // DDM prediction
            torch::Tensor inputn_new = normalizer->transform_X(input);
            torch::Tensor an = net->forward(inputn_new.unsqueeze(0));
            torch::Tensor BPoutput = normalizer->inverse_transform_Y(an);
            double V_DDM = BPoutput[0][0].item<double>();

            // SEDM prediction
            int idx = numTimeStepsTrain + w - 1 + n;
            double V_SEM = SEDM(input_params, tt[idx], Pc[idx], Pa[idx], T[idx], I[idx]);

            // Hybrid prediction
            double V_hybrid = (RR * V_SEM + V_DDM) / (RR + 1.0);

            aV_DDM.push_back(V_DDM);
            aV_SEM.push_back(V_SEM);
            aV_hybrid.push_back(V_hybrid);

            // Update input for next iteration
            if (n < num_test - 1) {
                input = input_test.index({torch::indexing::Slice(), n + 1}).clone();
                int dataset_row_size = static_cast<int>(input_columns.size()) + 1;
                int update_idx = dataset_row_size * w - dataset_row_size - 1;
                if (update_idx < num_features) {
                    input[update_idx] = V_hybrid;
                }
            }
        }

        // Calculate metrics
        std::cout << "\n=== Evaluation Results ===" << std::endl;

        double RR_SEM = calculateRSquared(YTest_vec, aV_SEM);
        double RR_DDM = calculateRSquared(YTest_vec, aV_DDM);
        double RR_Hybrid = calculateRSquared(YTest_vec, aV_hybrid);

        double RMSE_SEM = calculateRMSE(YTest_vec, aV_SEM);
        double RMSE_DDM = calculateRMSE(YTest_vec, aV_DDM);
        double RMSE_Hybrid = calculateRMSE(YTest_vec, aV_hybrid);

        double MAE_SEM = calculateMAE(YTest_vec, aV_SEM);
        double MAE_DDM = calculateMAE(YTest_vec, aV_DDM);
        double MAE_Hybrid = calculateMAE(YTest_vec, aV_hybrid);

        double RE_SEM = calculateMeanRE(YTest_vec, aV_SEM);
        double RE_DDM = calculateMeanRE(YTest_vec, aV_DDM);
        double RE_Hybrid = calculateMeanRE(YTest_vec, aV_hybrid);

        std::cout << "\nR2 Values:" << std::endl;
        std::cout << "  SEM:    " << RR_SEM << std::endl;
        std::cout << "  DDM:    " << RR_DDM << std::endl;
        std::cout << "  Hybrid: " << RR_Hybrid << std::endl;

        std::cout << "\nRMSE Values:" << std::endl;
        std::cout << "  SEM:    " << RMSE_SEM << std::endl;
        std::cout << "  DDM:    " << RMSE_DDM << std::endl;
        std::cout << "  Hybrid: " << RMSE_Hybrid << std::endl;

        std::cout << "\nMAE Values:" << std::endl;
        std::cout << "  SEM:    " << MAE_SEM << std::endl;
        std::cout << "  DDM:    " << MAE_DDM << std::endl;
        std::cout << "  Hybrid: " << MAE_Hybrid << std::endl;

        std::cout << "\nMean Relative Error (%):" << std::endl;
        std::cout << "  SEM:    " << RE_SEM << std::endl;
        std::cout << "  DDM:    " << RE_DDM << std::endl;
        std::cout << "  Hybrid: " << RE_Hybrid << std::endl;

        // Battery End-of-Life (EOL) estimation
        std::cout << "\n=== Battery End-of-Life (EOL) Estimate ===" << std::endl;
        double V_max = *std::max_element(aV_hybrid.begin(), aV_hybrid.end());
        double V_threshold = eol_threshold_ratio * V_max;
        int eol_index = -1;
        for (size_t i = 0; i < aV_hybrid.size(); ++i) {
            if (aV_hybrid[i] <= V_threshold) {
                eol_index = static_cast<int>(i);
                break;
            }
        }

        double eol_time = -1.0;
        double real_lifetime = -1.0;
        std::string eol_message;
        if (eol_index >= 0) {
            int data_idx = numTimeStepsTrain + w - 1 + eol_index;
            if (data_idx < static_cast<int>(tt.size())) {
                eol_time = tt[data_idx];
                real_lifetime = eol_time - time_begin;
            }
            eol_message = "EOL at step " + std::to_string(eol_index) +
                          " (time=" + std::to_string(eol_time) + "h, real_lifetime=" + std::to_string(real_lifetime) +
                          "h, V=" + std::to_string(aV_hybrid[eol_index]) + ")";
            std::cout << "Max predicted voltage: " << V_max << std::endl;
            std::cout << "80% threshold: " << V_threshold << std::endl;
            std::cout << "EOL occurs at prediction step " << eol_index
                    << " (data index " << data_idx << ", time = " << eol_time << " h)" << std::endl;
            if (time_begin > 0.0 && real_lifetime >= 0.0) {
                std::cout << "Real lifetime (after subtracting time_begin=" << time_begin << "): "
                        << real_lifetime << " h" << std::endl;
            }
        } else {
            eol_message = "No EOL crossing detected within prediction horizon";
            std::cout << "Max predicted voltage: " << V_max << std::endl;
            std::cout << "80% threshold: " << V_threshold << std::endl;
            std::cout << "No EOL crossing detected within prediction horizon." << std::endl;
        }

        // Save results
        std::ofstream outfile(output_predictions_path);
        outfile << "YTest,V_SEM,V_DDM,V_Hybrid,Error_SEM,Error_DDM,Error_Hybrid\n";
        for (size_t i = 0; i < YTest_vec.size(); ++i) {
            outfile << YTest_vec[i] << ","
                    << aV_SEM[i] << ","
                    << aV_DDM[i] << ","
                    << aV_hybrid[i] << ","
                    << (aV_SEM[i] - YTest_vec[i]) << ","
                    << (aV_DDM[i] - YTest_vec[i]) << ","
                    << (aV_hybrid[i] - YTest_vec[i]) << "\n";
        }
        outfile.close();
        std::cout << "\nResults saved to " << output_predictions_path << std::endl;

        controller.update_status("battery_lifespan", "completed", 0, 0, 0.0, RR_Hybrid,
                                 RMSE_Hybrid, MAE_Hybrid, eol_message);

        std::cout << "\nHybrid prediction completed!" << std::endl;
    }
} // namespace tju_torch

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
#include <omp.h>

namespace tju_torch {

// SEDM (Semi-Empirical Dynamic Model) function
static double SEDM(double tt, double Pc, double Pa, double T, double I) {
    const int nn = 300;
    const double A_cell = 190e-4;
    const double L_Pt = 4;
    const double F = 96487;
    const double R = 8.314472;
    const double P0 = 101325;
    const double Alpha_c = 0.2;
    const double Alpha_a = 0.8;
    const double Gamma_a = 0.5;
    const double Gamma_c = 1.0;
    const double c_o2_ref = 3.39;
    const double t_MEM = 15e-6;
    const double t_CLc = 15e-6;
    const double t_MPLc = 30e-6;
    const double t_GDLc = 180e-6;
    const double t_CHc = 440e-6;
    const double POR_CLc = 0.455;
    const double POR_MPLc = 0.4;
    const double POR_GDLc = 0.6;
    const double j_ref_a = 10;
    const double j_ref_c = 1e-5;

    const double b_leak = 1e-3;
    const double b_ECSA = -2e-4;
    const double b_ion = 2e-4;
    const double b_R = 1e-8;
    const double b_D = 1e-1;
    const double b_B = 1e-5;

    double r_leak = std::exp(b_leak * tt);
    double r_ECSA = std::exp(b_ECSA * tt);
    double r_ion = std::exp(b_ion * tt);

    double i_leak_ini = 20 * 190e-4;
    double A_ECSA_ini = 60 * (A_cell * L_Pt);
    double R_ion_ini = 100e-7 / A_cell;
    double R_ele_ini = 20e-7 / A_cell;
    double D_o2_ini = 2.652e-5 * std::pow(T / 333.15, 1.5) * (1.0 / Pc) * std::pow(POR_GDLc, 1.5);
    double K_c_ini = 100;

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
    double V_stack_sim = V_cell_sim * 300.0;

    return V_stack_sim;
}

void SEDMManager::run(const nlohmann::json& config) {
    std::cout << std::fixed << std::setprecision(6);

    // All file paths must be explicitly configured in JSON (no hard-coded defaults)
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

    int num_rows = config.value("num_rows", 900);
    int window_size = config.value("window_size", 5);
    int numTimeStepsTrain = config.value("train_samples", 300);
    int epochs = config.value("epochs", 1000);
    double goal_loss = config.value("goal_loss", 1e-10);
    double RR = config.value("rr", 4.0);

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

    TrainingController controller(control_file, status_file);
    std::string checkpoint_model = model_path + ".checkpoint.pt";
    std::string checkpoint_meta = model_path + ".checkpoint.json";

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA is available! Using GPU." << std::endl;
    } else {
        std::cout << "CUDA not available. Using CPU with OpenMP acceleration." << std::endl;
    }

    // Load data
    std::cout << "\nLoading data from " << data_file << "..." << std::endl;
    auto raw_data = readDataFile(data_file, num_rows);
    if (raw_data.empty()) {
        std::cerr << "Error: No data loaded!" << std::endl;
        controller.update_status("sedm", "stopped", 0, epochs, 0.0, 0.0, 0.0, 0.0, "Data load failed");
        return;
    }
    std::cout << "Loaded " << raw_data.size() << " rows of data." << std::endl;

    const int w = window_size;
    const size_t dd = raw_data.size();

    // Extract raw experimental data
    std::vector<double> tt, Pa, Pc, T, I, V_cell_exp;
    for (const auto& row : raw_data) {
        if (row.size() >= 12) {
            tt.push_back(row[0]);
            Pa.push_back(row[5]);
            Pc.push_back(row[4]);
            T.push_back(row[8] + 273.15);
            I.push_back(row[10]);
            V_cell_exp.push_back(row[11] / 300.0);
        }
    }

    // Prepare Input and Output for neural network
    std::vector<std::vector<double>> Input, Output;
    for (const auto& row : raw_data) {
        if (row.size() >= 12) {
            Input.push_back({row[4], row[5], row[8], row[10]});
            Output.push_back({row[11]});
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

    int num_samples = static_cast<int>(input_data_rows.size());
    int num_features = static_cast<int>(input_data_rows[0].size());
    int num_test = num_samples - numTimeStepsTrain;

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

    torch::Tensor input_test = torch::zeros({num_features, num_test});
    torch::Tensor output_test = torch::zeros({num_test});

    #pragma omp parallel for
    for (int i = 0; i < num_test; ++i) {
        for (int j = 0; j < num_features; ++j) {
            input_test[j][i] = input_data_rows[numTimeStepsTrain + i][j];
        }
        output_test[i] = output_data_vec[numTimeStepsTrain + i];
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
    torch::Tensor inputn = normalizer->transform_X(input_train);
    torch::Tensor outputn = normalizer->transform_Y(output_train);

    std::cout << "Using normalization: " << normalizer->name() << std::endl;

    // Load or train neural network
    std::cout << "\n=== Loading/Training Neural Network ===" << std::endl;
    auto net = std::make_shared<FeedForwardNet>(num_features, hidden_neurons, 1);
    net->to(device);

    bool model_loaded = false;
    try {
        torch::load(net, model_path);
        std::cout << "Loaded pre-trained model from " << model_path << std::endl;
        model_loaded = true;
    } catch (...) {
        std::cout << "No pre-trained model found. Training new model..." << std::endl;
    }

    if (!model_loaded) {
        std::string optimizer_type = config.value("optimizer_type", "lbfgs");
        torch::Tensor X_train = inputn.transpose(0, 1);
        torch::Tensor Y_train = outputn.unsqueeze(1);

        net->train();
        controller.update_status("sedm", "running", 0, epochs, 0.0, 0.0, 0.0, 0.0, "Training DDM model");

        if (optimizer_type == "lbfgs") {
            auto lbfgs_config = config["optimizer"]["lbfgs"];
            double lr = lbfgs_config.value("learning_rate", 1.0);
            torch::optim::LBFGS optimizer(
                net->parameters(),
                torch::optim::LBFGSOptions(lr)
                    .max_iter(lbfgs_config.value("max_iter", 20))
                    .max_eval(lbfgs_config.value("max_eval", 25))
                    .tolerance_grad(lbfgs_config.value("tolerance_grad", 1e-7))
                    .tolerance_change(lbfgs_config.value("tolerance_change", 1e-9))
                    .history_size(lbfgs_config.value("history_size", 100))
            );

            for (int epoch = 0; epoch < epochs; ++epoch) {
                // Check control commands during training
                std::string cmd = controller.read_command();
                if (cmd == "pause") {
                    controller.acknowledge_command();
                    controller.update_status("sedm", "paused", epoch, epochs, 0.0, 0.0, 0.0, 0.0, "Paused by user");
                    torch::save(net, checkpoint_model);
                    nlohmann::json meta = {{"epoch", epoch}, {"best_r2", 0.0}};
                    controller.save_checkpoint_meta(checkpoint_meta, meta);
                    std::string resume_cmd = controller.wait_for_resume();
                    if (resume_cmd == "stop") {
                        controller.update_status("sedm", "stopped", epoch, epochs, 0.0, 0.0, 0.0, 0.0, "Stopped by user");
                        return;
                    }
                    if (resume_cmd == "restart") {
                        controller.clear_checkpoint(checkpoint_meta, checkpoint_model);
                        controller.update_status("sedm", "running", 0, epochs, 0.0, 0.0, 0.0, 0.0, "Restarting");
                        epoch = -1;
                        continue;
                    }
                    controller.update_status("sedm", "running", epoch, epochs, 0.0, 0.0, 0.0, 0.0, "Resumed");
                }
                if (cmd == "stop") {
                    controller.acknowledge_command();
                    torch::save(net, checkpoint_model);
                    nlohmann::json meta = {{"epoch", epoch}, {"best_r2", 0.0}};
                    controller.save_checkpoint_meta(checkpoint_meta, meta);
                    controller.update_status("sedm", "stopped", epoch, epochs, 0.0, 0.0, 0.0, 0.0, "Stopped by user");
                    return;
                }
                if (cmd == "restart") {
                    controller.acknowledge_command();
                    controller.clear_checkpoint(checkpoint_meta, checkpoint_model);
                    controller.update_status("sedm", "running", 0, epochs, 0.0, 0.0, 0.0, 0.0, "Restarting");
                    epoch = -1;
                    continue;
                }

                auto closure = [&]() -> torch::Tensor {
                    optimizer.zero_grad();
                    torch::Tensor output = net->forward(X_train);
                    torch::Tensor loss = torch::mse_loss(output, Y_train);
                    loss.backward();
                    return loss;
                };

                torch::Tensor loss = optimizer.step(closure);

                if ((epoch + 1) % 100 == 0) {
                    std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss.item<double>() << std::endl;
                }
                if (loss.item<double>() < goal_loss) {
                    std::cout << "Converged at epoch " << (epoch + 1) << std::endl;
                    break;
                }

                if ((epoch + 1) % 100 == 0) {
                    controller.update_status("sedm", "running", epoch + 1, epochs,
                                             loss.item<double>(), 0.0, 0.0, 0.0,
                                             "Training DDM");
                }
            }
        } else {
            std::cerr << "SEDM fallback training only supports LBFGS currently." << std::endl;
        }

        torch::save(net, model_path);
        controller.clear_checkpoint(checkpoint_meta, checkpoint_model);
        std::cout << "Model saved to " << model_path << std::endl;
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

    controller.update_status("sedm", "running", epochs, epochs, 0.0, 0.0, 0.0, 0.0,
                             "Hybrid prediction in progress");

    for (int n = 0; n < num_test; ++n) {
        if ((n + 1) % 50 == 0) {
            std::cout << "Processing step " << (n + 1) << "/" << num_test << std::endl;
            // Check for stop command during long prediction
            std::string cmd = controller.read_command();
            if (cmd == "stop") {
                controller.acknowledge_command();
                controller.update_status("sedm", "stopped", epochs, epochs, 0.0, 0.0, 0.0, 0.0,
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
        double V_SEM = SEDM(tt[idx], Pc[idx], Pa[idx], T[idx], I[idx]);

        // Hybrid prediction
        double V_hybrid = (RR * V_SEM + V_DDM) / (RR + 1.0);

        aV_DDM.push_back(V_DDM);
        aV_SEM.push_back(V_SEM);
        aV_hybrid.push_back(V_hybrid);

        // Update input for next iteration
        if (n < num_test - 1) {
            input = input_test.index({torch::indexing::Slice(), n + 1}).clone();
            int update_idx = 5 * w - 5 - 1;
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

    controller.update_status("sedm", "completed", epochs, epochs, 0.0, RR_Hybrid,
                             RMSE_Hybrid, MAE_Hybrid, "Hybrid prediction completed");

    std::cout << "\nHybrid prediction completed!" << std::endl;
}

} // namespace tju_torch

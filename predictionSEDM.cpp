#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <omp.h>

// Feedforward Neural Network (same as prediction_model_FFN.cpp)
struct FeedForwardNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    FeedForwardNet(int64_t input_size, int64_t hidden1_size, int64_t hidden2_size, int64_t output_size) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden1_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden1_size, hidden2_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden2_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::sigmoid(fc1->forward(x));
        x = torch::sigmoid(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};

// Min-Max normalization class
class MinMaxScaler {
public:
    torch::Tensor x_min, x_max, y_min, y_max;

    void fit(const torch::Tensor& X, const torch::Tensor& Y) {
        x_min = std::get<0>(X.min(1));
        x_max = std::get<0>(X.max(1));
        y_min = Y.min();
        y_max = Y.max();
    }

    torch::Tensor transform_X(const torch::Tensor& X) {
        auto x_range = x_max - x_min;
        x_range = torch::where(x_range == 0, torch::ones_like(x_range), x_range);

        if (X.dim() == 2) {
            return 2.0 * (X - x_min.unsqueeze(1)) / x_range.unsqueeze(1) - 1.0;
        } else {
            return 2.0 * (X - x_min) / x_range - 1.0;
        }
    }

    torch::Tensor transform_Y(const torch::Tensor& Y) {
        auto y_range = y_max - y_min;
        if (y_range.item<double>() == 0) y_range = torch::ones_like(y_range);
        return 2.0 * (Y - y_min) / y_range - 1.0;
    }

    torch::Tensor inverse_transform_Y(const torch::Tensor& Y_norm) {
        auto y_range = y_max - y_min;
        if (y_range.item<double>() == 0) y_range = torch::ones_like(y_range);
        return (Y_norm + 1.0) * y_range / 2.0 + y_min;
    }
};

// SEDM (Semi-Empirical Dynamic Model) function
double SEDM(double tt, double Pc, double Pa, double T, double I) {
    // Model constants
    const int nn = 300;
    const double A_cell = 190e-4;  // m2
    const double L_Pt = 4;         // g m-2
    const double F = 96487;        // Faraday's Constant (coulomb/mole)
    const double R = 8.314472;     // Ideal gas constant (J/K/mol)
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

    // Degradation coefficients
    const double b_leak = 1e-3;
    const double b_ECSA = -2e-4;
    const double b_ion = 2e-4;
    const double b_R = 1e-8;
    const double b_D = 1e-1;
    const double b_B = 1e-5;

    // Degradation factors
    double r_leak = std::exp(b_leak * tt);
    double r_ECSA = std::exp(b_ECSA * tt);
    double r_ion = std::exp(b_ion * tt);

    // Initial parameters
    double i_leak_ini = 20 * 190e-4;
    double A_ECSA_ini = 60 * (A_cell * L_Pt);
    double R_ion_ini = 100e-7 / A_cell;
    double R_ele_ini = 20e-7 / A_cell;
    double D_o2_ini = 2.652e-5 * std::pow(T / 333.15, 1.5) * (1.0 / Pc) * std::pow(POR_GDLc, 1.5);
    double K_c_ini = 100;

    // Current parameters with degradation
    double i_leak = i_leak_ini * r_leak;
    double A_ECSA = A_ECSA_ini * r_ECSA;
    double R_total = R_ion_ini * r_ion + (R_ele_ini + b_R * tt);
    double D_o2 = D_o2_ini + b_D * tt;
    double K_c = K_c_ini + b_B * tt;

    // Nernst voltage
    double E_nernst = 1.229 - 0.846e-3 * (T - 298.15) +
                      R * T / 2.0 / F * (std::log(Pa) + 0.5 * std::log(Pc * 0.21));

    // Activation overpotential - anode
    double b_a = R * T / (2.0 * Alpha_a * F);
    double theta_T_a = std::exp(-1400.0 * (1.0 / T - 1.0 / 298.15));
    double c_h2_CLa = Pa * P0 / R / T;
    double k_ele_a = j_ref_a * std::pow(c_h2_CLa / c_o2_ref, Gamma_a) * theta_T_a;
    double V_act_a = b_a * (i_leak + I) / A_ECSA / k_ele_a;

    // Activation overpotential - cathode
    double b_c = R * T / (4.0 * Alpha_c * F);
    double theta_T_c = std::exp(-7900.0 * (1.0 / T - 1.0 / 298.15));
    double c_o2_CLc = 0.21 * Pc * P0 / R / T;
    double k_ele_c = j_ref_c * std::pow(c_o2_CLc / c_o2_ref, Gamma_c) * theta_T_c;
    double V_act_c = -b_c * std::log((i_leak + I) / A_ECSA / k_ele_c);

    // Ohmic overpotential
    double V_ohm = -I * R_total;

    // Concentration overpotential
    double D_o2_GDLc = 2.652e-5 * std::pow(T / 333.15, 1.5) * (1.0 / Pc) * std::pow(POR_GDLc, 1.5);
    double P_o2 = Pc * 0.21 * P0;
    double I_lim = 4.0 * F * (D_o2_GDLc / t_GDLc) * (P_o2 / R / T);
    double term_c = 1.0 - (I / A_ECSA) / I_lim;
    double V_conc_c = K_c * b_c * std::log(term_c);

    // Cell voltage
    double V_cell_sim = E_nernst + V_act_a + V_act_c + V_ohm + V_conc_c;

    // Stack voltage (300 cells)
    double V_stack_sim = V_cell_sim * 300.0;

    return V_stack_sim;
}

// Read data file
std::vector<std::vector<double>> readDataFile(const std::string& filename, int numRows) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        double value;

        while (ss >> value) {
            row.push_back(value);
        }

        if (!row.empty()) {
            data.push_back(row);
        }

        if (numRows > 0 && data.size() >= numRows) {
            break;
        }
    }

    file.close();
    return data;
}

// Calculate R-squared
double calculateRSquared(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double y_mean = 0.0;
    for (double val : y_true) y_mean += val;
    y_mean /= y_true.size();

    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_tot += std::pow(y_true[i] - y_mean, 2);
        ss_res += std::pow(y_true[i] - y_pred[i], 2);
    }

    return 1.0 - ss_res / ss_tot;
}

// Calculate RMSE
double calculateRMSE(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        sum += std::pow(y_true[i] - y_pred[i], 2);
    }
    return std::sqrt(sum / y_true.size());
}

// Calculate mean relative error
double calculateMeanRE(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        sum += std::abs((y_pred[i] - y_true[i]) / y_true[i]) * 100.0;
    }
    return sum / y_true.size();
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Set OpenMP threads
    int num_threads = omp_get_max_threads();

    // Allow user to override via environment variable
    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    if (omp_env != nullptr) {
        num_threads = std::atoi(omp_env);
    }

    omp_set_num_threads(num_threads);
    std::cout << "OpenMP enabled with " << num_threads << " threads." << std::endl;

    // Configure PyTorch to use OpenMP for intra-op parallelism
    torch::set_num_threads(num_threads);
    torch::set_num_interop_threads(num_threads);
    std::cout << "PyTorch configured to use " << num_threads << " threads." << std::endl;

    // Check for CUDA
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA is available! Using GPU." << std::endl;
    } else {
        std::cout << "CUDA not available. Using CPU with OpenMP acceleration." << std::endl;
    }

    // Load data
    std::cout << "\nLoading data from Data_V13_40kW.txt..." << std::endl;
    auto raw_data = readDataFile("Data_V13_40kW.txt", 900);

    if (raw_data.empty()) {
        std::cerr << "Error: No data loaded!" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << raw_data.size() << " rows of data." << std::endl;

    const int w = 5;
    const size_t dd = raw_data.size();

    // Extract raw experimental data
    std::vector<double> tt, Pa, Pc, T, I, V_cell_exp;
    for (const auto& row : raw_data) {
        if (row.size() >= 12) {
            tt.push_back(row[0]);              // Accumulated time (h)
            Pa.push_back(row[5]);              // Anode pressure
            Pc.push_back(row[4]);              // Cathode pressure
            T.push_back(row[8] + 273.15);      // Temperature (K)
            I.push_back(row[10]);              // Current (A)
            V_cell_exp.push_back(row[11] / 300.0); // Cell voltage (V)
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

    // Create Dataset
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

    int num_samples = input_data_rows.size();
    int num_features = input_data_rows[0].size();

    // Split train/test
    const int numTimeStepsTrain = 300;
    int num_test = num_samples - numTimeStepsTrain;

    // Create tensors
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
    MinMaxScaler scaler;
    scaler.fit(input_train, output_train);
    torch::Tensor inputn = scaler.transform_X(input_train);
    torch::Tensor outputn = scaler.transform_Y(output_train);

    // Load or train neural network
    std::cout << "\n=== Loading/Training Neural Network ===" << std::endl;
    auto net = std::make_shared<FeedForwardNet>(num_features, 50, 50, 1);
    net->to(device);

    // Try to load pre-trained model
    bool model_loaded = false;
    try {
        torch::load(net, "best_model.pt");
        std::cout << "Loaded pre-trained model from best_model.pt" << std::endl;
        model_loaded = true;
    } catch (...) {
        std::cout << "No pre-trained model found. Training new model..." << std::endl;
    }

    if (!model_loaded) {
        // Train the model
        torch::optim::LBFGS optimizer(
            net->parameters(),
            torch::optim::LBFGSOptions(1.0)
                .max_iter(20)
                .max_eval(25)
                .tolerance_grad(1e-7)
                .tolerance_change(1e-9)
                .history_size(100)
        );

        torch::Tensor X_train = inputn.transpose(0, 1);
        torch::Tensor Y_train = outputn.unsqueeze(1);

        net->train();
        for (int epoch = 0; epoch < 1000; ++epoch) {
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

            if (loss.item<double>() < 1e-10) {
                std::cout << "Converged at epoch " << (epoch + 1) << std::endl;
                break;
            }
        }

        torch::save(net, "best_model.pt");
        std::cout << "Model saved to best_model.pt" << std::endl;
    }

    // Hybrid prediction (DDM + SEDM)
    std::cout << "\n=== Hybrid Prediction (DDM + SEDM) ===" << std::endl;
    net->eval();
    torch::NoGradGuard no_grad;

    const double RR = 4.0;  // Weighting ratio
    std::vector<double> aV_DDM, aV_SEM, aV_hybrid;
    std::vector<double> YTest_vec;

    for (int i = 0; i < num_test; ++i) {
        YTest_vec.push_back(output_test[i].item<double>());
    }

    torch::Tensor input = input_test.index({torch::indexing::Slice(), 0}).clone();

    for (int n = 0; n < num_test; ++n) {
        if ((n + 1) % 50 == 0) {
            std::cout << "Processing step " << (n + 1) << "/" << num_test << std::endl;
        }

        // DDM prediction (neural network)
        torch::Tensor inputn_new = scaler.transform_X(input);
        torch::Tensor an = net->forward(inputn_new.unsqueeze(0));
        torch::Tensor BPoutput = scaler.inverse_transform_Y(an);
        double V_DDM = BPoutput[0][0].item<double>();

        // SEDM prediction
        int idx = numTimeStepsTrain + w - 1 + n;
        double V_SEM = SEDM(tt[idx], Pc[idx], Pa[idx], T[idx], I[idx]);

        // Hybrid prediction with dynamic weighting
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

    double RE_SEM = calculateMeanRE(YTest_vec, aV_SEM);
    double RE_DDM = calculateMeanRE(YTest_vec, aV_DDM);
    double RE_Hybrid = calculateMeanRE(YTest_vec, aV_hybrid);

    std::cout << "\nR² Values:" << std::endl;
    std::cout << "  SEM:    " << RR_SEM << std::endl;
    std::cout << "  DDM:    " << RR_DDM << std::endl;
    std::cout << "  Hybrid: " << RR_Hybrid << std::endl;

    std::cout << "\nRMSE Values:" << std::endl;
    std::cout << "  SEM:    " << RMSE_SEM << std::endl;
    std::cout << "  DDM:    " << RMSE_DDM << std::endl;
    std::cout << "  Hybrid: " << RMSE_Hybrid << std::endl;

    std::cout << "\nMean Relative Error (%):" << std::endl;
    std::cout << "  SEM:    " << RE_SEM << std::endl;
    std::cout << "  DDM:    " << RE_DDM << std::endl;
    std::cout << "  Hybrid: " << RE_Hybrid << std::endl;

    // Save results
    std::ofstream outfile("hybrid_predictions.csv");
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
    std::cout << "\nResults saved to hybrid_predictions.csv" << std::endl;

    std::cout << "\nHybrid prediction completed!" << std::endl;
    return 0;
}

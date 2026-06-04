#ifndef SEDM_MANAGER_H
#define SEDM_MANAGER_H

#include <nlohmann/json.hpp>

namespace tju_torch {

struct sedmInputParameter {
    int nn = 300;
    double A_cell = 0.019;
    double t_MEM = 15e-6;
    double t_CLc = 15e-6;
    double t_MPLc = 30e-6;
    double t_GDLc = 180e-6;
    double t_CHc = 440e-6;
    double POR_CLc = 0.455;
    double POR_MPLc = 0.4;
    double POR_GDLc = 0.6;
    double Alpha_a = 0.8;
    double Alpha_c = 0.2;
    double j_ref_a = 10.0;
    double j_ref_c = 1e-5;
    double K_c_ini = 100.0;
    double b_leak = 1e-3;
    double b_ECSA = -2e-4;
    double b_ion = 2e-4;
    double b_R = 1e-8;
    double b_D = 0.1;
    double b_B = 1e-5;
};

sedmInputParameter parse_sedm_input_parameter(const nlohmann::json& config);

class BatteryLifespanManager {
public:
    void run(const nlohmann::json& config);

private:
    void runTrain(const nlohmann::json& config);
    void runPredict(const nlohmann::json& config);
};

} // namespace tju_torch

#endif // SEDM_MANAGER_H

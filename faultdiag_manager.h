#ifndef FAULTDIAG_MANAGER_H
#define FAULTDIAG_MANAGER_H

#include <nlohmann/json.hpp>

namespace tju_torch {

class FaultDiagManager {
public:
    void run(const nlohmann::json& config);
};

} // namespace tju_torch

#endif // FAULTDIAG_MANAGER_H

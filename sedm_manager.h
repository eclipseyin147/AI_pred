#ifndef SEDM_MANAGER_H
#define SEDM_MANAGER_H

#include <nlohmann/json.hpp>

namespace tju_torch {

class SEDMManager {
public:
    void run(const nlohmann::json& config);
};

} // namespace tju_torch

#endif // SEDM_MANAGER_H

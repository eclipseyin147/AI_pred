#ifndef FFN_MANAGER_H
#define FFN_MANAGER_H

#include <nlohmann/json.hpp>

namespace tju_torch {

class FFNManager {
public:
    void run(const nlohmann::json& config);
};

} // namespace tju_torch

#endif // FFN_MANAGER_H

#include "training_controller.h"
#include <filesystem>

namespace tju_torch {

TrainingController::TrainingController(const std::string& control_file,
                                       const std::string& status_file)
    : control_file_(control_file),
      status_file_(status_file),
      last_command_("") {}

std::string TrainingController::read_command() {
    nlohmann::json j;
    if (!atomic_read_json(control_file_, j)) {
        return "";
    }
    if (!j.contains("command")) {
        return "";
    }
    std::string cmd = j.value("command", "");

    if (cmd == last_command_) {
        if (j.contains("timestamp_ms")) {
            // Always process - frontend should change timestamp for new commands
        } else {
            return "";
        }
    }
    return cmd;
}

void TrainingController::acknowledge_command() {
    nlohmann::json j;
    if (atomic_read_json(control_file_, j)) {
        last_command_ = j.value("command", "");
    }
    j["command"] = "";
    atomic_write_json(control_file_, j);
}

std::string TrainingController::wait_for_resume(int poll_interval_ms) {
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
        std::string cmd = read_command();
        if (cmd == "resume" || cmd == "stop" || cmd == "restart") {
            acknowledge_command();
            return cmd;
        }
    }
}

void TrainingController::write_status(const nlohmann::json& status) {
    atomic_write_json(status_file_, status);
}

int64_t TrainingController::current_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void TrainingController::merge_status_patch(const nlohmann::json& patch) {
    nlohmann::json current = nlohmann::json::object();
    atomic_read_json(status_file_, current);

    for (auto it = patch.begin(); it != patch.end(); ++it) {
        const std::string& key = it.key();
        if (it.value().is_object() && current.contains(key) && current[key].is_object()) {
            for (auto nested = it.value().begin(); nested != it.value().end(); ++nested) {
                current[key][nested.key()] = nested.value();
            }
        } else {
            current[key] = it.value();
        }
    }
    write_status(current);
}

void TrainingController::update_train_status(const std::string& mode,
                                             const std::string& state,
                                             int epoch,
                                             int total_epochs,
                                             double loss,
                                             double best_r2,
                                             double rmse,
                                             double mae,
                                             const std::string& message,
                                             const std::string& submode) {
    nlohmann::json patch = {
        {"mode", mode},
        {"state", state},
        {"submode", submode},
        {"message", message},
        {"timestamp_ms", current_timestamp_ms()},
        {"train", {
            {"epoch", epoch},
            {"total_epochs", total_epochs},
            {"loss", loss},
            {"best_r2", best_r2},
            {"rmse", rmse},
            {"mae", mae}
        }}
    };
    merge_status_patch(patch);
}

void TrainingController::update_predict_status(const std::string& mode,
                                               const std::string& state,
                                               const std::string& message,
                                               const nlohmann::json* eol) {
    nlohmann::json patch = {
        {"mode", mode},
        {"state", state},
        {"submode", "predict"},
        {"message", message},
        {"timestamp_ms", current_timestamp_ms()}
    };
    if (eol != nullptr) {
        patch["predict"] = nlohmann::json{{"eol", *eol}};
    }
    merge_status_patch(patch);
}

void TrainingController::save_checkpoint_meta(const std::string& meta_path,
                                              const nlohmann::json& meta) {
    atomic_write_json(meta_path, meta);
}

bool TrainingController::load_checkpoint_meta(const std::string& meta_path,
                                              nlohmann::json& meta) {
    return atomic_read_json(meta_path, meta);
}

void TrainingController::clear_checkpoint(const std::string& meta_path,
                                          const std::string& model_checkpoint_path) {
    try {
        if (std::filesystem::exists(meta_path)) {
            std::filesystem::remove(meta_path);
        }
        if (!model_checkpoint_path.empty() && std::filesystem::exists(model_checkpoint_path)) {
            std::filesystem::remove(model_checkpoint_path);
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: failed to clear checkpoint: " << e.what() << std::endl;
    }
}

bool TrainingController::checkpoint_exists(const std::string& meta_path) const {
    return std::filesystem::exists(meta_path);
}

void TrainingController::atomic_write_json(const std::string& path,
                                           const nlohmann::json& j) {
    try {
        std::string temp_path = path + ".tmp";
        {
            std::ofstream ofs(temp_path);
            if (ofs.is_open()) {
                ofs << j.dump(2);
            }
        }
        std::filesystem::rename(temp_path, path);
    } catch (const std::exception& e) {
        std::cerr << "Warning: failed to write JSON to " << path << ": " << e.what() << std::endl;
    }
}

bool TrainingController::atomic_read_json(const std::string& path,
                                          nlohmann::json& j) {
    try {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            return false;
        }
        ifs >> j;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

} // namespace tju_torch

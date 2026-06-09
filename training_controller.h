#ifndef TRAINING_CONTROLLER_H
#define TRAINING_CONTROLLER_H

#include <nlohmann/json.hpp>
#include <string>
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>

namespace tju_torch {

// ============================================================================
// TrainingController
// Provides file-based IPC for Qt frontend to control training:
//   - Reads commands from control.json (pause/resume/stop/restart)
//   - Writes status to status.json (merged by task block: train / predict)
//   - Saves/loads checkpoint metadata JSON for restart support
// ============================================================================
class TrainingController {
public:
    TrainingController(const std::string& control_file,
                       const std::string& status_file);

    // ------------------------------------------------------------------------
    // Control file
    // ------------------------------------------------------------------------
    std::string read_command();
    void acknowledge_command();
    std::string wait_for_resume(int poll_interval_ms = 500);

    // ------------------------------------------------------------------------
    // Status file (atomic write: temp -> rename, merge with existing)
    // ------------------------------------------------------------------------
    void write_status(const nlohmann::json& status);

    // Patch top-level fields + "train" block; leaves "predict" block unchanged.
    void update_train_status(const std::string& mode,
                             const std::string& state,
                             int epoch,
                             int total_epochs,
                             double loss,
                             double best_r2,
                             double rmse,
                             double mae,
                             const std::string& message,
                             const std::string& submode = "train");

    // Patch top-level fields + "predict" block; leaves "train" block unchanged.
    void update_predict_status(const std::string& mode,
                               const std::string& state,
                               const std::string& message,
                               const nlohmann::json* eol = nullptr);

    // ------------------------------------------------------------------------
    // Checkpoint metadata
    // ------------------------------------------------------------------------
    void save_checkpoint_meta(const std::string& meta_path,
                              const nlohmann::json& meta);
    bool load_checkpoint_meta(const std::string& meta_path,
                              nlohmann::json& meta);
    void clear_checkpoint(const std::string& meta_path,
                          const std::string& model_checkpoint_path = "");

    bool checkpoint_exists(const std::string& meta_path) const;

private:
    std::string control_file_;
    std::string status_file_;
    std::string last_command_;

    void merge_status_patch(const nlohmann::json& patch);
    static int64_t current_timestamp_ms();
    void atomic_write_json(const std::string& path,
                           const nlohmann::json& j);
    bool atomic_read_json(const std::string& path,
                          nlohmann::json& j);
};

} // namespace tju_torch

#endif // TRAINING_CONTROLLER_H

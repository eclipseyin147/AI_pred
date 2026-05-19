# TJU-Torch Project Guide for AI Coding Agents

## Project Overview

This is an academic research project that ports MATLAB-based deep learning models to C++ using LibTorch (PyTorch C++ API). The project focuses on two main tasks:

1. **Fuel Cell Lifespan Prediction** — Feedforward Neural Network (FFN) regression to predict stack voltage degradation, combined with a Semi-Empirical Dynamic Model (SEDM) for hybrid forecasting.
2. **Fault Diagnosis** — Sequence classification using 1D CNN and Temporal Convolutional Networks (TCN) on multi-channel time-series data.

The project name is `tju-torch` (CMake project name). It was created by `siqi` and targets C++17.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | C++17 |
| Build System | CMake (minimum 3.10) |
| Package Manager | vcpkg |
| ML Framework | LibTorch (PyTorch C++) |
| Parallelism | OpenMP |
| JSON Config | nlohmann-json |
| MAT File I/O | matio (for MATLAB `.mat` files) |
| Post-Processing | Python 3 (NumPy, Pandas, Matplotlib) |
| Reference Implementations | MATLAB (R202x) |

### vcpkg Dependencies (`vcpkg.json`)

- `libtorch` (with CUDA feature)
- `cudnn`
- `nlohmann-json`
- `matio`

---

## Project Structure

```
.
├── CMakeLists.txt              # Main CMake configuration
├── vcpkg.json                  # vcpkg manifest
├── vcpkg-configuration.json    # vcpkg overlay ports config
├── unified_config.json         # Unified config for single executable
├── config.json                 # Legacy FFN training config (kept for reference)
├── config_adamw.json           # Legacy FFN config
├── config_rmsprop.json         # Legacy FFN config
├── faultDiag_config.json       # Legacy fault diagnosis config
├── faultDiag_config_cnn.json   # Legacy CNN config
├── faultDiag_config_tcn.json   # Legacy TCN config
├── faultDiag_config_test.json  # Legacy test config
│
├── unified_main.cpp            # Unified executable entry point
├── common_ffn.h                # Shared FFN, normalizers, metrics
├── ffn_manager.h/.cpp          # FFN training manager class
├── sedm_manager.h/.cpp         # SEDM hybrid prediction manager class
├── faultdiag_manager.h/.cpp    # Fault diagnosis manager class
├── training_controller.h/.cpp  # Qt IPC control/status module
│
├── faultDiagnosis.cpp          # CNN/TCN model implementations
├── faultDiagnosis.h            # Core ML classes and trainers
│
├── prediction_model_FFN.cpp    # Legacy standalone FFN (preserved, not built)
├── predictionSEDM.cpp          # Legacy standalone SEDM (preserved, not built)
├── faultDiagMain.cpp           # Legacy fault diagnosis CLI (preserved, not built)
│
├── Prediction_model_1.m        # MATLAB reference: FFN training
├── Prediction_model_2.m        # MATLAB reference: SEDM hybrid prediction
├── m1DCNN.m                    # MATLAB reference: 1D CNN fault diagnosis
├── myselfTCN.m                 # MATLAB reference: TCN fault diagnosis
│
├── post_process.py             # Python visualization & evaluation report
│
├── Data_V13_40kW.txt           # Raw fuel cell experimental data (900 rows)
├── ALL_Traindata1.mat          # TCN training data (MATLAB .mat)
├── Copy_of_select-model fault data.mat   # CNN training data
├── net1.mat                    # Pre-trained MATLAB network
├── best_model.pt               # Saved LibTorch model checkpoint
│
├── FIXES_APPLIED.md            # Critical bug fixes log (MATLAB alignment)
├── MAT_FILE_STRUCTURE.md       # MAT file format documentation
├── cmake-build-relwithdebinfo/ # CMake build directory (Ninja generator)
└── install/bin/                # Install target for executables
```

---

## Build Instructions

### Prerequisites

- Windows (primary target; Linux/Mac supported in CMake)
- CMake >= 3.10
- vcpkg (with environment variable `VCPKG_ROOT` set)
- Visual Studio 2019+ or MinGW-w64 (Windows)
- CUDA toolkit (optional; CPU-only fallback works)

### Configure and Build

```powershell
# From project root
cmake -B cmake-build-relwithdebinfo -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo `-DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build cmake-build-relwithdebinfo
```

The project uses the **Ninja** generator by default in the existing build directory.

### Install

```powershell
cmake --install cmake-build-relwithdebinfo --prefix install
```

On Windows, this also copies required Torch DLLs (`c10.dll`, `torch_cpu.dll`, etc.) to `install/bin/`.

### Build Outputs

A single unified executable is produced (legacy executables are preserved in source but no longer built):

| Executable | Source Files | Purpose |
|------------|--------------|---------|
| `tju-torch` | `unified_main.cpp`, `ffn_manager.cpp`, `sedm_manager.cpp`, `faultdiag_manager.cpp`, `faultDiagnosis.cpp`, `training_controller.cpp` | Unified entry point dispatching to FFN, SEDM, or fault diagnosis via JSON `mode` |

**Legacy sources** (`prediction_model_FFN.cpp`, `predictionSEDM.cpp`, `faultDiagMain.cpp`) are kept in the repository for reference but removed from the CMake build target.

---

## Runtime Architecture

The unified executable `tju-torch.exe` reads a single JSON config file and dispatches to the appropriate manager class based on the top-level `mode` field.

```powershell
.\tju-torch.exe [config_file.json]
```

If no config file is provided, defaults to `unified_config.json`.

### 1. FFN Mode (`"mode": "ffn"`)

Managed by `FFNManager` (`ffn_manager.cpp`).
- **Input**: Plain-text data file (path from JSON `input_data_path`)
- **Preprocessing**: Sliding window with configurable `window_size`, pluggable normalization
- **Model**: Generalized FFN with configurable hidden layer count and neuron counts
- **Optimizers Supported**: LBFGS, Adam, AdamW, RMSprop
- **Outputs**: Model `.pt`, `predictions.csv`, `training_log.csv`, `status.json`
- **Metrics**: R², RMSE, MAE
- **Control**: Supports pause/resume/stop/restart via `control.json`

### 2. SEDM Mode (`"mode": "sedm"`)

Managed by `SEDMManager` (`sedm_manager.cpp`).
- Loads pre-trained model (path from JSON `model_path`) or trains fallback
- Runs DDM neural network alongside physics-based SEDM
- Combines predictions: `V_hybrid = (RR * V_SEM + V_DDM) / (RR + 1)`
- **Outputs**: `hybrid_predictions.csv`, `status.json`
- **Metrics**: R², RMSE, MAE for SEM / DDM / Hybrid
- **Control**: Supports pause/resume/stop/restart via `control.json`

### 3. FaultDiag Mode (`"mode": "faultdiag"`)

Managed by `FaultDiagManager` (`faultdiag_manager.cpp`).
- **Input**: MATLAB `.mat` files (path from JSON `input_mat_path`)
- **Submodes** (`submode`): `cnn`, `tcn`, `test`
- **Outputs**: Saved model `.pt`, confusion matrix, accuracy metrics, `status.json`
- **Control**: Supports pause/resume/stop/restart via `control.json`

---

---

## Configuration System

The unified executable uses a single JSON configuration file. The top-level `mode` field selects the task. All file paths **must be explicitly provided** in the JSON — there are no hard-coded defaults.

### Unified Config Schema (`unified_config.json`)

```json
{
  "mode": "ffn",
  "ffn": {
    "input_data_path": "Data_V13_40kW.txt",
    "output_model_path": "ffn_best_model.pt",
    "output_predictions_path": "predictions.csv",
    "output_training_log_path": "training_log.csv",
    "control_file_path": "control.json",
    "status_file_path": "status.json",
    "hidden_layers": 2,
    "hidden_layer_neurons": [50, 50],
    "learning_rate": 0.001,
    "epochs": 15000,
    "batch_size": 32,
    "optimizer_type": "adamw",
    "optimizer": { "adamw": { "learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-10, "weight_decay": 0.001 } },
    "normalization": { "enabled": true, "method": "minmax_neg1_1" },
    "goal_loss": 2e-5,
    "max_iterations": 1000,
    "target_r2": 0.85,
    "print_interval": 200,
    "window_size": 5,
    "train_samples": 300,
    "num_rows": 900,
    "input_columns": [4, 5, 8, 10],
    "output_column": 11
  },
  "sedm": {
    "input_data_path": "Data_V13_40kW.txt",
    "model_path": "sedm_best_model.pt",
    "output_predictions_path": "hybrid_predictions.csv",
    "control_file_path": "control.json",
    "status_file_path": "status.json",
    "hidden_layers": 2,
    "hidden_layer_neurons": [50, 50],
    "learning_rate": 1.0,
    "epochs": 1000,
    "batch_size": 32,
    "optimizer_type": "lbfgs",
    "optimizer": { "lbfgs": { "learning_rate": 1.0, "max_iter": 20, "max_eval": 25, "tolerance_grad": 1e-7, "tolerance_change": 1e-9, "history_size": 100 } },
    "normalization": { "enabled": true, "method": "minmax_neg1_1" },
    "goal_loss": 1e-10,
    "window_size": 5,
    "train_samples": 300,
    "num_rows": 900,
    "rr": 4.0,
    "input_columns": [4, 5, 8, 10],
    "output_column": 11,
    "time_column": 0
  },
  "faultdiag": {
    "submode": "tcn",
    "input_mat_path": "ALL_Traindata1.mat",
    "output_model_path": "fault_best_model.pt",
    "control_file_path": "control.json",
    "status_file_path": "status.json",
    "hidden_layers": 2,
    "hidden_layer_neurons": [64, 48],
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 26,
    "optimizer": "adam",
    "use_gpu": false,
    "data_var": "AXTrain3",
    "label_var": "AYTrain",
    "val_data_var": "AXTest3",
    "val_label_var": "AYTest",
    "train_split": 0.8,
    "validation_frequency": 10,
    "normalization": { "enabled": true, "method": "rescale_symmetric" },
    "cnn_filter_size": 2,
    "cnn_num_filters": 32,
    "tcn_num_blocks": 4,
    "tcn_num_filters": 64,
    "tcn_filter_size": 3,
    "tcn_dropout": 0.005
  }
}
```

#### Normalization Methods

| Method String | Description |
|---------------|-------------|
| `minmax_neg1_1` | Min-Max to [-1, 1] (MATLAB `mapminmax` style) |
| `minmax_0_1` | Min-Max to [0, 1] |
| `z_score` | StandardScaler: (x - mean) / std |
| `none` / `disabled` | No normalization |

For fault diagnosis, `rescale_symmetric` is an alias for `minmax_neg1_1`.

#### Column Selection (txt/csv modes)

For `ffn` and `sedm` modes, the input features and output target are selected by column index (0-based) instead of hard-coded positions:

| Field | Type | Mode | Description |
|-------|------|------|-------------|
| `input_columns` | `int[]` | `ffn`, `sedm` | 0-based column indices to use as neural-network input features |
| `output_column` | `int` | `ffn`, `sedm` | 0-based column index to use as the prediction target |
| `time_column` | `int` | `sedm` | 0-based column index for the time variable used by the SEDM physics model |

If these fields are omitted, the defaults match the original hard-coded behavior:
- `input_columns`: `[4, 5, 8, 10]`
- `output_column`: `11`
- `time_column`: `0`

> **SEDM note**: The SEDM physics model expects `input_columns` to contain at least 4 columns mapping to `[Pc, Pa, T, I]` in that order.

#### Required File Path Fields (per mode)

| Mode | Required Path Fields |
|------|---------------------|
| `ffn` | `input_data_path`, `output_model_path`, `output_predictions_path`, `output_training_log_path`, `control_file_path`, `status_file_path` |
| `sedm` | `input_data_path`, `model_path`, `output_predictions_path`, `control_file_path`, `status_file_path` |
| `faultdiag` | `input_mat_path`, `output_model_path`, `control_file_path`, `status_file_path` |

---

## Code Organization & Module Divisions

### `faultDiagnosis.h` / `faultDiagnosis.cpp`

The core ML library lives in the `lifespanPred` namespace.

| Class/Struct | Purpose |
|--------------|---------|
| `Conv1DNet` | 1D CNN: 2×(Conv1d → ReLU → LayerNorm) → GlobalMaxPool → Linear |
| `TCNBlock` | Dilated causal convolution block with residual connection and manual spatial dropout |
| `TCNNet` | Stack of `TCNBlock`s (dilations 1,2,4,8) → FC layers |
| `SequenceNormalizer` | Min-max rescale-symmetric normalization to `[-1, 1]` |
| `SequenceDataset` | torch::data::Dataset for variable-length sequences |
| `SequenceTrainer<ModelType>` | Template trainer with Adam/SGD/RMSprop, validation, best-model saving |
| `SequenceCollate` | Left-padding collate function for mini-batches |
| `load_mat_data()` | Parse `.mat` cell arrays or 2D/3D matrices into `vector<Tensor>` |

**Important**: `faultDiagnosis.cpp` contains the bulk of the implementation. `faultDiagnosis.h` contains template method implementations inline (required for C++ templates).

### `prediction_model_FFN.cpp`

Self-contained single-file executable. Defines:
- `FeedForwardNet` (3-layer, sigmoid hidden)
- `MinMaxScaler` (column-wise normalization)
- `readDataFile()` (whitespace-delimited text parser)
- Training loop with optimizer dispatch

### `predictionSEDM.cpp`

Self-contained single-file executable. Defines:
- `FeedForwardNet` and `MinMaxScaler` (duplicated from FFN)
- `SEDM()` function — physics-based fuel cell voltage model with 300 cells
- Hybrid prediction loop that iteratively updates inputs with hybrid output

---

## Critical Development Conventions

### MATLAB → C++ Alignment

This project is a **faithful port** of MATLAB reference code. Many design decisions exist solely to match MATLAB behavior:

1. **Label Indexing**: MATLAB uses 1-based class labels; C++ converts to 0-based in `load_mat_data()`.
2. **Data Layout**: MATLAB is column-major. Tensors are stored as `[features, sequence_length]` to match MATLAB cell arrays.
3. **Normalization**:
   - **CNN**: NO input normalization (`normalize_input = false`) — matches `m1DCNN.m`
   - **TCN**: rescale-symmetric to `[-1, 1]` (`normalize_input = true`) — matches `myselfTCN.m`
4. **Padding Direction**: Left padding only (`torch::cat({padding, seq}, 1)`) to match MATLAB `SequencePaddingDirection="left"`.
5. **CNN Layer Order**: Conv → ReLU → LayerNorm (not Conv → LayerNorm → ReLU). This was a critical fix documented in `FIXES_APPLIED.md`.
6. **TCN Causal Padding**: Only left-side padding with `(kernel_size - 1) * dilation`. No symmetric padding.
7. **Sliding Window**: FFN models concatenate `w=5` rows horizontally then drop the last element, matching MATLAB exactly.

### OpenMP & Threading

- All three executables set OpenMP threads via `omp_set_num_threads()`.
- PyTorch intra-op and inter-op threads are set to match: `torch::set_num_threads()`, `torch::set_num_interop_threads()`.
- Environment variable `OMP_NUM_THREADS` overrides the default.

### Memory & CUDA Safety

- The code explicitly calls `.contiguous()` on tensors before/after `permute()` and before residual additions. This prevents CUDA runtime errors with non-contiguous tensors.
- CUDA is auto-detected but falls back to CPU seamlessly.

---

## Testing Strategies

There is **no automated unit test framework** (no GoogleTest, Catch2, etc.). Testing is manual and integration-level:

1. **Smoke Test**: Run `faultDiag.exe faultDiag_config_test.json` to train CNN and TCN on synthetic random sequences.
2. **MATLAB Comparison**: Train CNN/TCN in C++ and compare confusion matrices and accuracies with MATLAB outputs.
3. **R² Validation**: FFN training stops only when R² exceeds a threshold; manual inspection of `predictions.csv` is required.
4. **Python Post-Processing**: Run `post_process.py` after `SEDM.exe` to verify hybrid model metrics and generate plots.

### Validation Checklist

- [ ] `faultDiag` CNN mode runs without normalization messages.
- [ ] `faultDiag` TCN mode prints "Input normalization: ENABLED".
- [ ] `FFNPredictor` produces `predictions.csv` with RMSE and R² > target.
- [ ] `SEDM` produces `hybrid_predictions.csv` and Python plots.
- [ ] Confusion matrices from C++ match MATLAB reference values.

---

## Deployment & Distribution

- The CMake `install()` target copies executables to `install/bin/`.
- On Windows, required Torch DLLs are also installed automatically.
- Models are saved as `.pt` files (LibTorch serialized modules) and are portable across CPU/GPU builds.
- No containerization (Docker) is present.

---

## Security Considerations

- **No input sanitization** on JSON config files — malformed configs will crash with `nlohmann::json` exceptions or `std::runtime_error`.
- **No bounds checking** on `.mat` file dimensions beyond basic null checks.
- **File paths** are constructed with simple string concatenation (`cwd.u8string() + "/faultDiag_config.json"`).
- **No authentication or networking** — all local file I/O.

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `matio library not found` | vcpkg manifest not applied | Ensure `-DCMAKE_TOOLCHAIN_FILE` points to vcpkg.cmake |
| CUDA out of memory | TCN batch size too large for GPU | Set `"use_gpu": false` in JSON or reduce `batch_size` |
| Non-contiguous tensor error | Missing `.contiguous()` after permute | Already fixed; do not remove `.contiguous()` calls |
| Label mismatch | Forgot 1-based → 0-based conversion | Ensure `load_mat_data()` subtracts 1 from labels |
| CNN accuracy lower than MATLAB | Layer order or normalization wrong | Verify `Conv→ReLU→LN` and `normalize_input=false` |

---

## Qt Frontend Integration

The unified executable `tju-torch.exe` is designed to be controlled by a Qt desktop application via **JSON file-based IPC**. No network sockets or shared memory are required — Qt simply writes a control file and periodically reads a status file.

### Launching the Executable from Qt

```cpp
#include <QProcess>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTimer>

// 1. Start the process
QProcess *process = new QProcess(parent);
process->setProgram("tju-torch.exe");
process->setArguments({"unified_config.json"});  // or any config file
process->start();

// 2. Poll status.json periodically (e.g., every 500 ms)
QTimer *timer = new QTimer(parent);
connect(timer, &QTimer::timeout, [=]() {
    QFile file("status.json");
    if (!file.open(QIODevice::ReadOnly)) return;
    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    QJsonObject status = doc.object();

    int epoch = status["epoch"].toInt();
    int total = status["total_epochs"].toInt();
    QString state = status["state"].toString();  // "running", "paused", "stopped", "completed"
    double loss = status["loss"].toDouble();
    double r2   = status["best_r2"].toDouble();
    double rmse = status["rmse"].toDouble();
    double mae  = status["mae"].toDouble();
    QString msg = status["message"].toString();

    // Update UI progress bars, labels, etc.
});
timer->start(500);
```

### Control File (`control.json`)

Qt writes commands to the control file. The executable polls this file once per epoch.

```json
{
  "command": "pause",
  "timestamp_ms": 1715432100123
}
```

| Command | Effect |
|---------|--------|
| `run` | Default state; no action if already running |
| `pause` | Pause training after current epoch; save checkpoint |
| `resume` | Resume from paused state |
| `stop` | Gracefully stop training; save checkpoint and exit |
| `restart` | Clear checkpoint files and restart training from scratch |

**Qt Example — Pause Button:**
```cpp
void MainWindow::onPauseClicked() {
    QFile file("control.json");
    if (file.open(QIODevice::WriteOnly)) {
        QJsonObject obj;
        obj["command"] = "pause";
        obj["timestamp_ms"] = QDateTime::currentDateTimeUtc().toMSecsSinceEpoch();
        file.write(QJsonDocument(obj).toJson(QJsonDocument::Compact));
        file.close();
    }
}
```

### Status File (`status.json`)

The executable writes its current state after every epoch (and immediately upon state changes). Qt should read this file periodically.

```json
{
  "mode": "ffn",
  "state": "running",
  "epoch": 150,
  "total_epochs": 1000,
  "loss": 0.00123,
  "best_r2": 0.92,
  "rmse": 0.045,
  "mae": 0.032,
  "message": "Training iteration 2, epoch 150",
  "timestamp_ms": 1715432101000
}
```

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `"ffn"`, `"sedm"`, or `"faultdiag"` |
| `state` | string | `"idle"`, `"running"`, `"paused"`, `"stopped"`, `"completed"` |
| `epoch` | int | Current epoch number |
| `total_epochs` | int | Total epochs configured |
| `loss` | double | Current training loss |
| `best_r2` | double | Best R² achieved so far (FFN/SEDM) |
| `rmse` | double | Current RMSE |
| `mae` | double | Current MAE |
| `message` | string | Human-readable status message |
| `timestamp_ms` | int64 | Monotonic timestamp for freshness check |

### Checkpoint System

When `pause` or `stop` is issued, the executable automatically saves:
- **Model checkpoint**: `<output_model_path>.checkpoint.pt`
- **Metadata checkpoint**: `<output_model_path>.checkpoint.json`

The metadata JSON contains:
```json
{
  "iteration": 2,
  "epoch": 150,
  "best_r2": 0.92,
  "hidden_layer_neurons": [50, 50]
}
```

On startup, if a checkpoint exists **and** no `restart` command is pending, the executable automatically resumes from the checkpoint. To force a fresh start, send `restart` before launching (or delete the `.checkpoint.pt` and `.checkpoint.json` files).

### Thread Safety & Atomic Writes

Both control and status files are written using **atomic rename** (write to `.tmp`, then `std::filesystem::rename`). This prevents race conditions where Qt and the C++ process access the file simultaneously. Qt does not need file locking — simply open, read/write, and close quickly.

### File Path Configuration

All IPC file paths are **configurable via JSON** (`control_file_path`, `status_file_path`). The Qt application and the JSON config must agree on these paths. No paths are hard-coded in the executable.

---

## File Reference for Agents

- **To modify model architecture**: Edit `faultDiagnosis.h` (declarations) and `faultDiagnosis.cpp` (implementations).
- **To modify FFN training**: Edit `ffn_manager.h` and `ffn_manager.cpp`.
- **To modify hybrid prediction**: Edit `sedm_manager.h` and `sedm_manager.cpp`.
- **To modify fault diagnosis CLI**: Edit `faultdiag_manager.h` and `faultdiag_manager.cpp`.
- **To modify Qt IPC behavior**: Edit `training_controller.h` and `training_controller.cpp`.
- **To add a new optimizer**: Update the `SequenceTrainer` constructor in `faultDiagnosis.h` and the optimizer dispatch in `ffn_manager.cpp` / `sedm_manager.cpp`.
- **To change data preprocessing**: Update `common_ffn.h` (`DataNormalizer` hierarchy) or `SequenceNormalizer` in `faultDiagnosis.h` / `faultDiagnosis.cpp`.

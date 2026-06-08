# TJU-Torch Project Guide for AI Coding Agents

## Project Overview

This is an academic research project that ports MATLAB-based deep learning models to C++ using LibTorch (PyTorch C++ API). The project focuses on two main tasks:

1. **Fuel Cell Lifespan Prediction** — Battery lifespan prediction using a Feedforward Neural Network (FFN) for time-series regression, combined with a Semi-Empirical Dynamic Model (SEDM) for hybrid forecasting. Supports both training (`train`) and prediction (`predict`) sub-modes.
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
├── data_reader.h               # Unified txt/csv data file reader (header skip, row range)
├── sedm_manager.h/.cpp         # Battery lifespan manager (train + predict submodes)
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
| `tju-torch` | `unified_main.cpp`, `sedm_manager.cpp`, `faultdiag_manager.cpp`, `faultDiagnosis.cpp`, `training_controller.cpp` | Unified entry point dispatching via CLI `--mode` / `--submode` |

**Legacy sources** (`prediction_model_FFN.cpp`, `predictionSEDM.cpp`, `faultDiagMain.cpp`) are kept in the repository for reference but removed from the CMake build target.

---

## Runtime Architecture

The unified executable `tju-torch.exe` loads `./unified_config.json` from the **current working directory** (GUI sets cwd to the case root). The run task is selected via **command-line** `mode` / `submode`; JSON retains persistent parameters only.

```powershell
cd /path/to/case
.\tju-torch.exe --mode battery_lifespan --submode train
.\tju-torch.exe --mode battery_lifespan --submode predict
.\tju-torch.exe --mode faultdiag
.\tju-torch.exe --mode battery_lifespan --submode train --config unified_config.json
```

| CLI flag | Values | Notes |
|----------|--------|-------|
| `--mode` / `-m` | `battery_lifespan`, `faultdiag` | CLI only; default `battery_lifespan` |
| `--submode` / `-s` | `train`, `predict` | CLI only (lifespan); default `train` |
| `--config` / `-c` | path | Default `./unified_config.json` relative to cwd |

JSON does **not** contain root `mode` or `battery_lifespan.submode`. Omitted CLI flags use defaults `battery_lifespan` / `train`.

### 1. Battery Lifespan Mode (`--mode battery_lifespan`)

Managed by `BatteryLifespanManager` (`sedm_manager.cpp`).
- **Submode `train`**:
  - Trains a DDM neural network on plain-text or CSV time-series data (via `data_reader.h`).
  - Supports **AdamW** and **LBFGS** optimizers.
  - LBFGS includes automatic seed-retry (seeds 42–51) to handle divergence.
  - Iterative outer loop validates best R² on test data (compatible with old FFN training style).
  - **Outputs**: Model `.pt`, `battery_predictions.csv` (`Time,YTest,YPred,Error`), `battery_training_log.csv`, `status.json`
  - **Metrics**: R², RMSE, MAE
- **Submode `predict`**:
  - Loads a pre-trained model (path from JSON `model_path`).
  - Runs DDM neural network alongside physics-based SEDM.
  - Combines predictions: `V_hybrid = (RR * V_SEM + V_DDM) / (RR + 1)`.
  - Computes **battery end-of-life (EOL)**: first time point on the full hybrid curve where `V_Hybrid <= eol_threshold_ratio * V_max` (default 80%).
  - **Outputs**: `battery_predictions.csv` (`Time,YTest,V_SEM,V_DDM,V_Hybrid,Error_SEM,Error_DDM,Error_Hybrid`), `status.json`
  - **Metrics**: EOL estimate only (R²/RMSE/MAE are printed during `train` validation, not in `predict`)
- **Control**: Supports pause/resume/stop/restart via `control.json` for both submodes.

### 3. FaultDiag Mode (`"mode": "faultdiag"`)

Managed by `FaultDiagManager` (`faultdiag_manager.cpp`).
- **Input**: MATLAB `.mat` files (path from JSON `input_mat_path`)
- **Submodes** (`submode`): `cnn`, `tcn`, `test`
- **Outputs**: Saved model `.pt`, confusion matrix, accuracy metrics, `status.json`
- **Control**: Supports pause/resume/stop/restart via `control.json`

---

---

## Configuration System

The unified executable uses `./unified_config.json` in the current working directory. **Run intent** (`mode`, lifespan `submode`) comes from the command line; JSON holds data paths, hyperparameters, and physics settings. All file paths **must be explicitly provided** in the JSON — there are no hard-coded defaults for paths.

### Unified Config Schema (`unified_config.json`)

```json
{
  "battery_lifespan": {
    "input_data_path": "Data_V13_40kW.txt",
    "model_path": "battery_best_model.pt",
    "output_predictions_path": "battery_predictions.csv",
    "output_training_log_path": "battery_training_log.csv",
    "control_file_path": "control.json",
    "status_file_path": "status.json",
    "hidden_layers": 2,
    "hidden_layer_neurons": [50, 50],
    "learning_rate": 1.0,
    "epochs": 1000,
    "batch_size": 32,
    "optimizer_type": "adamw",
    "optimizer": {
      "lbfgs": { "learning_rate": 1.0, "max_iter": 20, "max_eval": 25, "tolerance_grad": 1e-7, "tolerance_change": 1e-9, "history_size": 100 },
      "adamw": { "learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.001 }
    },
    "normalization": { "enabled": true, "method": "minmax_neg1_1" },
    "goal_loss": 1e-10,
    "max_iterations": 1000,
    "target_r2": 0.85,
    "print_interval": 200,
    "window_size": 5,
    "training_sample_ratio": 0.5,
    "num_rows_begin": 1,
    "num_rows_end": 900,
    "time_begin": 0.0,
    "eol_threshold_ratio": 0.80,
    "rr": 4.0,
    "input_columns": [4, 5, 8, 10],
    "output_column": 11,
    "time_column": 0,
    "nn": 300,
    "A_cell": 0.019,
    "t_MEM": 0.000015,
    "t_CLc": 0.000015,
    "t_MPLc": 0.00003,
    "t_GDLc": 0.00018,
    "t_CHc": 0.00044,
    "POR_CLc": 0.455,
    "POR_MPLc": 0.4,
    "POR_GDLc": 0.6,
    "Alpha_a": 0.8,
    "Alpha_c": 0.2,
    "j_ref_a": 10.0,
    "j_ref_c": 0.00001,
    "K_c_ini": 100.0,
    "b_leak": 0.001,
    "b_ECSA": -0.0002,
    "b_ion": 0.0002,
    "b_R": 1e-8,
    "b_D": 0.1,
    "b_B": 0.00001
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

For `battery_lifespan` mode, input data files may be **whitespace-delimited txt** or **comma-separated csv** (auto-detected; first non-numeric line treated as header and skipped). Reading is implemented in `data_reader.h` and used via `common_ffn.h`.

For `battery_lifespan` mode, the input features and output target are selected by column index (0-based) instead of hard-coded positions:

| Field | Type | Mode | Description |
|-------|------|------|-------------|
| `input_columns` | `int[]` | `battery_lifespan` | 0-based column indices to use as neural-network input features |
| `output_column` | `int` | `battery_lifespan` | 0-based column index to use as the prediction target |
| `time_column` | `int` | `battery_lifespan` | 0-based column index for the time variable used by the SEDM physics model |

| `num_rows_begin` | `int` | `battery_lifespan` | First numeric data row to read, **1-based inclusive** (matches GUI row numbers; header lines are not counted). Values `<= 0` are treated as `1`. |
| `num_rows_end` | `int` | `battery_lifespan` | Last numeric data row to read, **1-based inclusive**. Use `-1` or `<= 0` to read through the last data row. |
| `time_begin` | `double` | `battery_lifespan` | Starting time offset (in hours) written by the GUI. When calculating EOL, this value is subtracted from the absolute time to obtain the real lifetime. Default is `0.0`. |
| `eol_threshold_ratio` | `double` | `battery_lifespan` | Ratio of maximum predicted hybrid voltage used as the EOL threshold. `V_max` is taken from the full prediction curve; EOL is the **first** time point where `V_Hybrid <= eol_threshold_ratio * V_max`. Default is `0.80`.

> **Backward compatibility**: The legacy `num_rows` field is still supported. If `num_rows_begin`/`num_rows_end` are not present but `num_rows` is, the behavior defaults to `num_rows_begin=1` and `num_rows_end=num_rows` (first `num_rows` data rows). A legacy `num_rows_begin` of `0` is treated as `1`.

If these fields are omitted, the defaults match the original hard-coded behavior:
- `input_columns`: `[4, 5, 8, 10]`
- `output_column`: `11`
- `time_column`: `0`

> **SEDM note**: The SEDM physics model (used in `predict` submode) expects `input_columns` to contain at least 4 columns mapping to `[Pc, Pa, T, I]` in that order.

#### SEDM Input Parameters (flat keys under `battery_lifespan`)

Optional fields at the same level as `input_data_path`, `rr`, etc.; omitted keys use `sedmInputParameter` defaults in `sedm_manager.h`.

| Field | Default | Category | Description |
|-------|---------|----------|-------------|
| `nn` | `300` | Stack geometry | Number of cells; stack voltage = cell voltage × `nn` |
| `A_cell` | `0.019` | Stack geometry | Active area (m²) |
| `t_MEM`, `t_CLc`, `t_MPLc`, `t_GDLc`, `t_CHc` | see config | Stack geometry | Layer thicknesses (m); `t_GDLc` used in limit-current term |
| `POR_CLc`, `POR_MPLc`, `POR_GDLc` | see config | Stack geometry | Porosities; `POR_GDLc` used in diffusion terms |
| `Alpha_a`, `Alpha_c` | `0.8`, `0.2` | Initialization | Charge-transfer coefficients |
| `j_ref_a`, `j_ref_c` | `10.0`, `1e-5` | Initialization | Reference exchange current densities |
| `K_c_ini` | `100.0` | Initialization | Initial concentration-loss scaling |
| `b_leak`, `b_ECSA`, `b_ion`, `b_R`, `b_D`, `b_B` | see config | Degradation | Time-dependent degradation factors |

Constants fixed in code (not in JSON): `F`, `R`, `P0`, `Gamma_a/c`, `c_o2_ref`, `L_Pt`, and derived `i_leak_ini` (= `20 * A_cell`), `A_ECSA_ini`, `R_ion_ini`, `R_ele_ini`.

#### Required File Path Fields (per mode)

| Mode | Required Path Fields |
|------|---------------------|
| `battery_lifespan` (train) | `input_data_path`, `model_path`, `output_predictions_path`, `output_training_log_path`, `control_file_path`, `status_file_path` |
| `battery_lifespan` (predict) | `input_data_path`, `model_path`, `output_predictions_path`, `control_file_path`, `status_file_path` |
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
- Legacy local `readDataFile()` (whitespace txt only; **not** used by unified `tju-torch` build)
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
4. **Python Post-Processing**: Run `post_process.py` after `battery_lifespan predict` to verify hybrid model metrics and generate plots.

### Validation Checklist

- [ ] `faultDiag` CNN mode runs without normalization messages.
- [ ] `faultDiag` TCN mode prints "Input normalization: ENABLED".
- [ ] `battery_lifespan train` produces `predictions.csv` with RMSE and R² > target.
- [ ] `battery_lifespan predict` produces `hybrid_predictions.csv` and Python plots.
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
process->setArguments({"--mode", "battery_lifespan", "--submode", "train", "--config", "unified_config.json"});
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
    QString submode = status["submode"].toString();  // "train" | "predict"
    QString state = status["state"].toString();
    QString msg = status["message"].toString();

    if (submode == "train" && status.contains("train")) {
        QJsonObject train = status["train"].toObject();
        int epoch = train["epoch"].toInt();
        int total = train["total_epochs"].toInt();
        double r2 = train["best_r2"].toDouble();
        // Update training progress UI...
    }
    if (status.contains("predict")) {
        QJsonObject predict = status["predict"].toObject();
        if (predict.contains("eol")) {
            QJsonObject eol = predict["eol"].toObject();
            if (eol["detected"].toBool()) {
                double x = eol["x"].toDouble();
                double y = eol["y"].toDouble();
                // Mark EOL point on chart...
            }
        }
    }
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

Status updates **merge** into the existing file by task block: `train` and `predict` are updated independently so one run does not wipe the other.

```json
{
  "mode": "battery_lifespan",
  "state": "completed",
  "submode": "predict",
  "message": "EOL at time=512.5h (...)",
  "train": {
    "epoch": 60,
    "total_epochs": 60,
    "loss": 0.0,
    "best_r2": 0.92,
    "rmse": 0.04,
    "mae": 0.03
  },
  "predict": {
    "eol": {
      "detected": true,
      "x": 512.5,
      "y": 155.2
    }
  },
  "timestamp_ms": 1715432101000
}
```

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `"battery_lifespan"` or `"faultdiag"` |
| `state` | string | `"idle"`, `"running"`, `"paused"`, `"stopped"`, `"completed"` |
| `submode` | string | Last active task: `"train"` or `"predict"` (`battery_lifespan` only) |
| `message` | string | Human-readable status for the current update |
| `train` | object | Training metrics; updated only by **train** / **faultdiag** runs |
| `predict` | object | Prediction results; updated only when **predict** completes (with `eol`) |
| `timestamp_ms` | int64 | Monotonic timestamp for freshness check |

#### `train` object

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | int | Current epoch |
| `total_epochs` | int | Total epochs configured |
| `loss` | double | Current training loss |
| `best_r2` | double | Best R² (`battery_lifespan` train) |
| `rmse` | double | RMSE |
| `mae` | double | MAE |

#### `predict.eol` object (`battery_lifespan` predict only)

| Field | Type | Description |
|-------|------|-------------|
| `detected` | bool | `true` if EOL crossing was found on the hybrid curve |
| `x` | double \| null | Abscissa: time (hours), same as CSV `Time` |
| `y` | double \| null | Ordinate: `V_Hybrid` at the crossing |

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
- **To modify battery lifespan training / prediction**: Edit `sedm_manager.h` and `sedm_manager.cpp`.
- **To modify fault diagnosis CLI**: Edit `faultdiag_manager.h` and `faultdiag_manager.cpp`.
- **To modify Qt IPC behavior**: Edit `training_controller.h` and `training_controller.cpp`.
- **To add a new optimizer**: Update the `SequenceTrainer` constructor in `faultDiagnosis.h` and the optimizer dispatch in `sedm_manager.cpp`.
- **To change data preprocessing**: Update `data_reader.h` / `common_ffn.h` (`DataNormalizer`, `readDataFile`) for battery lifespan txt/csv input, or `SequenceNormalizer` in `faultDiagnosis.h` / `faultDiagnosis.cpp` for fault diagnosis.

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
├── config.json                 # Default FFN training config (AdamW)
├── config_adamw.json           # FFN config (AdamW variant)
├── config_rmsprop.json         # FFN config (RMSprop variant)
├── faultDiag_config.json       # Default fault diagnosis config (CNN mode)
├── faultDiag_config_cnn.json   # CNN-specific config
├── faultDiag_config_tcn.json   # TCN-specific config
├── faultDiag_config_test.json  # Synthetic data test config
│
├── prediction_model_FFN.cpp    # FFN lifespan predictor (regression)
├── predictionSEDM.cpp          # SEDM + FFN hybrid predictor
├── faultDiagMain.cpp           # Fault diagnosis CLI entry point
├── faultDiagnosis.cpp          # CNN/TCN model implementations
├── faultDiagnosis.h            # Core ML classes and trainers
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

Three executables are produced:

| Executable | Source Files | Purpose |
|------------|--------------|---------|
| `FFNPredictor` | `prediction_model_FFN.cpp` | Train FFN for voltage prediction |
| `SEDM` | `predictionSEDM.cpp` | Hybrid SEDM+FFN prediction |
| `faultDiag` | `faultDiagMain.cpp`, `faultDiagnosis.cpp` | Train/evaluate CNN or TCN classifiers |

---

## Runtime Architecture

### 1. FFNPredictor (Regression)

- **Input**: Plain-text data file (`Data_V13_40kW.txt`)
- **Preprocessing**: Sliding window with `w=5`, min-max normalization to `[-1, 1]` (MATLAB `mapminmax` equivalent)
- **Model**: 3-layer FFN with sigmoid activations on hidden layers
- **Optimizers Supported**: LBFGS, Adam, AdamW, RMSprop (configured via JSON)
- **Output**: `best_model.pt`, `predictions.csv`, `training_log.csv`
- **Early Stopping**: Stops when R² > `target_r2` (default 0.85)

Run:
```powershell
.\FFNPredictor.exe [config.json]
```

### 2. SEDM (Hybrid Model)

- Loads pre-trained `best_model.pt` (or trains if missing)
- Runs the neural network (DDM) alongside the physics-based SEDM
- Combines predictions with dynamic weighting: `V_hybrid = (RR * V_SEM + V_DDM) / (RR + 1)` where `RR = 4`
- Outputs `hybrid_predictions.csv`
- Python post-processing generates charts and `evaluation_report.txt`

Run:
```powershell
.\SEDM.exe
python post_process.py
```

### 3. faultDiag (Sequence Classification)

- **Input**: MATLAB `.mat` files (via `matio` library)
- **Modes**:
  - `cnn` — 1D CNN with global max pooling
  - `tcn` — Temporal Convolutional Network with dilated causal convolutions
  - `test` — Synthetic data smoke test
- **Data Variables**: Configurable via JSON (e.g., `AXTrain3`, `AYTrain`)
- **Output**: Saved model `.pt`, confusion matrix, accuracy metrics

Run:
```powershell
.\faultDiag.exe [config_file.json]
```

Default config is `faultDiag_config.json`.

---

## Configuration System

All executables use **JSON configuration files**. Key schemas:

### FFN Config (`config.json`)
```json
{
  "optimizer": { "type": "adamw", "adamw": { "learning_rate": 0.001, ... } },
  "training": { "epochs": 15000, "goal_loss": 2e-5, "max_iterations": 1000, "target_r2": 0.85 },
  "model": { "hidden_layer1": 50, "hidden_layer2": 50 },
  "data": { "window_size": 5, "train_samples": 300, "data_file": "Data_V13_40kW.txt", "num_rows": 900 }
}
```

### Fault Diagnosis Config
```json
{
  "mode": "tcn",
  "data": { "mat_file": "ALL_Traindata1.mat", "data_var": "AXTrain3", "label_var": "AYTrain", "train_split": 0.8 },
  "training": { "epochs": 100, "batch_size": 26, "learning_rate": 0.001, "optimizer": "adam", "use_gpu": false },
  "model": { "cnn": { ... }, "tcn": { "num_blocks": 4, "num_filters": 64, "filter_size": 3, "dropout": 0.005 } },
  "output": { "model_save_path": "best_model.pt" }
}
```

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

## File Reference for Agents

- **To modify model architecture**: Edit `faultDiagnosis.h` (declarations) and `faultDiagnosis.cpp` (implementations).
- **To modify FFN training**: Edit `prediction_model_FFN.cpp`.
- **To modify hybrid prediction**: Edit `predictionSEDM.cpp`.
- **To modify fault diagnosis CLI**: Edit `faultDiagMain.cpp`.
- **To add a new optimizer**: Update the `SequenceTrainer` constructor in `faultDiagnosis.h` and `prediction_model_FFN.cpp`.
- **To change data preprocessing**: Update `SequenceNormalizer` or `MinMaxScaler` classes.

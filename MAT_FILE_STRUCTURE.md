# MAT File Structure Analysis

## 1. m1DCNN.m (1D CNN Model)

### MAT File Information:
- **File**: `Copy_of_select-model fault data.mat`
- **Training Data Variable**: `XTrain2`
- **Training Labels Variable**: `YTrain2`
- **Validation Data Variable**: `XText2`
- **Validation Labels Variable**: `YText2`

### Model Configuration:
- **Number of Classes**: 4
- **Number of Features**: 8
- **Filter Size**: 2
- **Number of Filters**: 32
- **Batch Size**: 26
- **Epochs**: 200
- **Optimizer**: Adam (lr=0.001)
- **Normalization**: None specified (default)
- **Padding Direction**: Left

### Data Structure:
```matlab
XTrain2: Cell array where each cell is a matrix [8, sequenceLength]
         - 8 features (numFeatures)
         - Variable sequence lengths
YTrain2: Numeric array [numSamples] with values 1-4
         - Converted to categorical in MATLAB
```

### Example Usage:
```json
{
  "mode": "cnn",
  "data": {
    "mat_file": "Copy_of_select-model fault data.mat",
    "data_var": "XTrain2",
    "label_var": "YTrain2",
    "train_split": 0.8
  }
}
```

---

## 2. myselfTCN.m (Temporal Convolutional Network)

### MAT File Information:
- **Training File**: `ALL_Traindata1.mat`
  - **Training Data Variable**: `AXTrain3`
  - **Training Labels Variable**: `AYTrain`

- **Testing File**: `3S_Testdata.mat`
  - **Testing Data Variable**: `AXTest`
  - **Testing Labels Variable**: `AYTest`

### Model Configuration:
- **Number of Classes**: 5
- **Number of Features**: Determined from data (`size(XTrain,1)`)
- **Number of Blocks**: 4
- **Number of Filters**: 64
- **Filter Size**: 3
- **Dropout**: 0.005
- **Batch Size**: 1
- **Epochs**: 100
- **Optimizer**: Adam
- **Normalization**: rescale-symmetric ([-1, 1])
- **Dilation Factors**: [1, 2, 4, 8] (2^(i-1) for i=1:4)

### Data Structure:
```matlab
AXTrain3: Cell array where each cell is a matrix [numFeatures, sequenceLength]
          - Variable number of features (extracted from data)
          - Variable sequence lengths
AYTrain:  Numeric array [numSamples] with values 1-5
          - Converted to categorical in MATLAB
```

### Example Usage:
```json
{
  "mode": "tcn",
  "data": {
    "mat_file": "ALL_Traindata1.mat",
    "data_var": "AXTrain3",
    "label_var": "AYTrain",
    "train_split": 0.8
  }
}
```

---

## Key Differences

| Aspect | 1D CNN | TCN |
|--------|--------|-----|
| **MAT File** | `Copy_of_select-model fault data.mat` | `ALL_Traindata1.mat` |
| **Data Variable** | `XTrain2` | `AXTrain3` |
| **Label Variable** | `YTrain2` | `AYTrain` |
| **Classes** | 4 | 5 |
| **Features** | 8 (fixed) | Variable (from data) |
| **Batch Size** | 26 | 1 |
| **Epochs** | 200 | 100 |
| **Normalization** | None | rescale-symmetric [-1,1] |
| **Architecture** | 2 Conv layers + Global Max Pool | 4 TCN blocks with residual |

---

## Data Format Requirements

### MATLAB Cell Array Format:
Both models expect data in MATLAB cell array format:

```matlab
% In MATLAB
XTrain = {
    [8×100 double],  % Sequence 1: 8 features, 100 time steps
    [8×150 double],  % Sequence 2: 8 features, 150 time steps
    [8×75 double],   % Sequence 3: 8 features, 75 time steps
    ...
};

YTrain = [1; 2; 1; 3; ...];  % Class labels (1-indexed)
```

### C++ libtorch Format:
The `load_mat_data()` function converts to:

```cpp
// Each sequence is a torch::Tensor [num_features, sequence_length]
std::vector<torch::Tensor> sequences;
sequences.push_back(torch::randn({8, 100}));  // Sequence 1
sequences.push_back(torch::randn({8, 150}));  // Sequence 2

// Labels are converted to 0-indexed
std::vector<int64_t> labels = {0, 1, 0, 2, ...};  // 0-indexed
```

---

## Important Notes

1. **Label Indexing**:
   - MATLAB uses 1-based indexing (labels: 1, 2, 3, 4, 5)
   - C++ uses 0-based indexing (labels: 0, 1, 2, 3, 4)
   - The `load_mat_data()` function automatically converts: `labels[i] = matlab_label - 1`

2. **Data Layout**:
   - MATLAB cell arrays store each sequence as `[features, time_steps]`
   - This is preserved in libtorch format
   - MATLAB is column-major, so proper indexing is critical

3. **Sequence Padding**:
   - Both MATLAB models use left padding
   - Implemented in the trainer with `torch::cat({padding, seq}, 1)`

4. **Normalization**:
   - CNN: No input normalization specified
   - TCN: Uses rescale-symmetric normalization to [-1, 1]
   - Implemented in `SequenceNormalizer::transform()`

5. **Variable-Length Sequences**:
   - Both models support variable-length sequences
   - Padding is applied during mini-batch creation
   - Trainer handles padding automatically

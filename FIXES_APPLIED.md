# Critical Fixes Applied to Match MATLAB Implementation

## Issues Found and Fixed:

### 1. **CNN Layer Order (CRITICAL FIX)**

**❌ Previous (INCORRECT):**
```cpp
Conv1D → LayerNorm → ReLU
Conv1D → LayerNorm → ReLU
```

**✅ Current (CORRECT - matches MATLAB):**
```cpp
Conv1D → ReLU → LayerNorm  // Line 47-51
Conv1D → ReLU → LayerNorm  // Line 54-58
```

**MATLAB Reference (m1DCNN.m:18-23):**
```matlab
convolution1dLayer → reluLayer → layerNormalizationLayer
convolution1dLayer → reluLayer → layerNormalizationLayer
```

---

### 2. **Input Normalization (CRITICAL FIX)**

**❌ Previous (INCORRECT):**
- Both CNN and TCN were normalized

**✅ Current (CORRECT):**
- **CNN**: NO normalization (normalize_input = false)
- **TCN**: rescale-symmetric [-1,1] normalization (normalize_input = true)

**MATLAB Reference:**
- m1DCNN.m:17 - `sequenceInputLayer(numFeatures,Name="input")`
  - NO normalization (rescale-symmetric is commented out)
- myselfTCN.m:53 - `sequenceInputLayer(numFeatures,Normalization="rescale-symmetric")`
  - HAS rescale-symmetric normalization

**Implementation (lifespannPred.h:231-248):**
```cpp
if (config.normalize_input) {
    normalizer.fit(train_sequences);
    norm_train_sequences = normalizer.transform(train_sequences);
} else {
    norm_train_sequences = train_sequences;  // No normalization
}
```

---

### 3. **Causal Padding (CRITICAL FIX)**

**❌ Previous (INCORRECT):**
```cpp
.padding(padding)  // Symmetric padding (both sides)
// Then trim from right
```
- Conv could see **future values** (not truly causal!)

**✅ Current (CORRECT - matches MATLAB):**
```cpp
.padding(0)  // No automatic padding
// Manual left-only padding in forward()
torch::nn::functional::pad(x, {left_pad, 0})  // LEFT pad only
```

**MATLAB Reference (myselfTCN.m:62):**
```matlab
convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
```
- `Padding="causal"` means LEFT padding only

**Implementation (lifespannPred.cpp:101-112):**
```cpp
// Causal padding: pad LEFT only with (filter_size - 1) * dilation
int64_t left_pad = (kernel_size - 1) * dilation_factor;
torch::nn::functional::pad(x, {left_pad, 0})  // left pad only
```

---

### 4. **Data Loading for 2D Matrix (FIX)**

**Issue:** MAT file `AXTrain3` is `[6, 9068]` (not a cell array)
- 6 features
- 9068 samples (each column is one sample)

**Solution (lifespannPred.cpp:460-493):**
```cpp
// 2D matrix: [features, num_samples]
// Treat each column as a separate sequence with seq_length=1
for (size_t s = 0; s < num_samples; ++s) {
    torch::Tensor tensor = torch::zeros({num_features, 1});
    for (size_t f = 0; f < num_features; ++f) {
        tensor[f][0] = data[f + num_features * s];  // Column-major indexing
    }
    sequences.push_back(tensor);  // Each sequence is [6, 1]
}
```

---

## Configuration Changes:

### TrainingConfig (lifespannPred.h:111-131)
**New field added:**
```cpp
bool normalize_input = false;  // Set to true for TCN, false for CNN
```

### Usage (lifespannMain.cpp):
```cpp
// CNN configuration
cnn_config.normalize_input = false;  // NO normalization

// TCN configuration
tcn_config.normalize_input = true;   // rescale-symmetric normalization
```

---

## Summary of Alignment with MATLAB:

| Feature | MATLAB m1DCNN.m | MATLAB myselfTCN.m | C++ Implementation |
|---------|-----------------|-------------------|-------------------|
| **Layer Order** | Conv→ReLU→LN | Conv→LN→ReLU | ✅ Matches both |
| **Input Norm** | None | rescale-symmetric | ✅ Conditional |
| **Padding** | "same" | "causal" | ✅ Correct |
| **Dilation** | N/A | [1,2,4,8] | ✅ Correct |
| **Skip Connection** | N/A | 1×1 conv (first) | ✅ Correct |
| **Data Format** | Cell array | Cell or Matrix | ✅ Handles both |

---

## Testing Recommendations:

1. **Test CNN with m1DCNN data:**
   ```bash
   lifespannPred faultDiag_config_cnn.json
   ```
   Expected: NO normalization message

2. **Test TCN with myselfTCN data:**
   ```bash
   lifespannPred faultDiag_config.json
   ```
   Expected: Normalization ENABLED message

3. **Verify outputs:**
   - Check confusion matrices
   - Compare accuracies with MATLAB results
   - Ensure sequence lengths are preserved

All critical issues are now fixed! ✅

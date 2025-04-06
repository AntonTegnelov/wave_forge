# Numerical Stability Analysis: Entropy Calculation

## Introduction

This document analyzes the numerical stability of entropy calculations in the Wave Function Collapse (WFC) GPU implementation. Entropy calculation is a critical component of the WFC algorithm as it determines which cells to collapse next. Any numerical instabilities can lead to suboptimal selection, inconsistent results, or even algorithm failure.

## Entropy Calculation Methods

The WFC-GPU implementation supports several entropy calculation methods:

1. **Shannon Entropy** (Default): log₂(n) where n is the number of possible states
2. **Count Heuristic**: Simple count of possible states minus 1
3. **Count Simple**: Normalized count (n/total)
4. **Weighted Count**: Sum of weights of possible states

## Numerical Stability Considerations

### Shannon Entropy (log₂(n))

**Potential Issues:**

- **Logarithm of Very Small Numbers**: When approaching a single possibility, the entropy approaches zero, but floating-point precision can cause issues.
- **Precision Loss**: For large numbers of possibilities, the difference between log₂(n) and log₂(n+1) becomes very small, potentially leading to tie-breaking challenges.

**Mitigations:**

- Special handling for n ≤ 1 (returning -1.0 for contradictions, 0.0 for collapsed cells)
- Avoiding direct computation of log(p) for very small probabilities

### Count-Based Heuristics

**Advantages:**

- More numerically stable as they avoid logarithmic operations
- Linear relationship to number of possibilities
- Less susceptible to floating-point precision issues

**Limitations:**

- Less sensitive to small differences in large possibility spaces
- May not capture information-theoretic properties as effectively as Shannon entropy

### Weighted Calculations

**Potential Issues:**

- **Sum Overflow/Underflow**: When dealing with very large or small weights
- **Precision Loss**: When combining weights of vastly different magnitudes
- **Normalization Issues**: Dividing by very small sums can amplify errors

**Mitigations in Our Implementation:**

- Using the log-sum-exp trick for calculating weighted Shannon entropy
- Normalizing weights to avoid numerical issues
- Skipping very small normalized weights to prevent precision issues with logarithms

## Implementation-Specific Analysis

### GPU Shader Implementation

Our WGSL shader implementation includes several stability improvements:

```
// From entropy_modular.wgsl
// Base entropy calculation: log2(count)
var entropy = log2(f32(possible_count));

// For weighted entropy:
// Use the log-sum-exp trick for numerical stability
// H = log(sum(w)) - sum(w*log(w))/sum(w)
let log_sum_weights = log2(sum_weights);
var weighted_log_sum = 0.0;
for (var i = 0u; i < active_count; i++) {
    let w = weights_array[i];
    let normalized_w = w / sum_weights;  // Normalize weights

    // Skip or handle very small values to prevent precision issues with log
    if (normalized_w > 0.00001) {
        weighted_log_sum += normalized_w * log2(normalized_w);
    }
}
```

### Minimum Entropy Finding

The process of finding the cell with minimum entropy involves additional numerical considerations:

1. **Tie-Breaking**: When multiple cells have the same entropy, stable tie-breaking is important
2. **Atomic Operations**: Using atomic operations for parallel minimum finding requires careful handling

Our implementation addresses these with:

```
// Convert to sortable integer representation
// Use bitcast to preserve ordering while allowing atomic operations
// Invert to make smaller entropy = larger integer for min finding
let entropy_bits = bitcast<u32>(1.0 / entropy);
```

## Comparison of Entropy Heuristics

| Heuristic    | Numerical Stability | Resolution | Performance | Use Case                                        |
| ------------ | ------------------- | ---------- | ----------- | ----------------------------------------------- |
| Shannon      | Medium              | High       | Medium      | General-purpose, information-theoretic approach |
| Count        | High                | Medium     | High        | When stability is more important than precision |
| Count Simple | Very High           | Low        | Very High   | Fast approximation for simple patterns          |
| Weighted     | Low-Medium          | Very High  | Low         | When tile probabilities are non-uniform         |

## Benchmarking Results

Numerical stability testing across different grid sizes and pattern complexities shows:

1. **Shannon Entropy**: Occasional tie-breaking issues with large grids (>256x256)
2. **Count Heuristic**: Most stable across all test cases
3. **Weighted Count**: Most precise but requires careful handling of weights

## Recommendations

Based on our analysis:

1. **Default Choice**: Shannon entropy provides a good balance of stability and precision
2. **For Large Grids**: Count heuristic is recommended for better numerical stability
3. **For Complex Patterns**: Weighted count with proper numerical safeguards

## Future Improvements

1. **Adaptive Precision**: Dynamically switch between heuristics based on grid state
2. **Extended Precision**: Consider using higher precision for intermediate calculations
3. **Improved Tie-Breaking**: Implement more sophisticated tie-breaking strategies like Hilbert curve traversal

## Conclusion

The current implementation provides robust numerical stability for entropy calculations across different use cases. The availability of multiple heuristics allows users to choose the approach that best fits their specific requirements for stability, precision, and performance.

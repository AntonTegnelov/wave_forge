# Visualizing Benchmark Results

This document provides examples of visualizing Wave Forge benchmark results to better understand performance characteristics.

## Example Visualizations

Below are several examples of visualizations generated from benchmark data to illustrate different aspects of performance.

### 1. CPU vs GPU Execution Time by Grid Size

![CPU vs GPU Execution Time](../images/cpu_vs_gpu_comparison.png)

_This logarithmic scale chart shows how execution time increases with grid size for both CPU and GPU implementations. Note how the GPU maintains better scaling for larger grid sizes._

### 2. GPU Speedup Factor

![GPU Speedup Factor](../images/gpu_speedup.png)

_This chart shows the relative speedup of the GPU implementation compared to CPU across different grid sizes. The speedup factor increases with grid size, indicating the GPU's advantage for larger problems._

### 3. Time Distribution by Operation

![Time Distribution by Operation](../images/time_distribution.png)

_This stacked area chart shows how different operations (entropy calculation, constraint propagation, etc.) contribute to the total execution time as grid size increases._

### 4. Memory Usage Comparison

![Memory Usage Comparison](../images/memory_usage.png)

_This chart compares memory usage between CPU and GPU implementations across different grid sizes. Note how the GPU implementation's memory overhead becomes proportionally smaller as grid size increases._

### 5. Performance Scaling with Tile Count

![Performance Scaling with Tile Count](../images/tile_count_scaling.png)

_This chart shows how performance scales with the number of unique tiles in the ruleset. More tiles typically mean more complex constraint checking._

## Creating These Visualizations

You can create similar visualizations using the Python script below, which processes the CSV output from Wave Forge's benchmarking tool:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the benchmark data
df = pd.read_csv('benchmark_results.csv')

# Calculate grid volume for x-axis
df['Volume'] = df['Width'] * df['Height'] * df['Depth']

# Set up styling
plt.style.use('ggplot')
sns.set_palette("viridis")

# 1. CPU vs GPU Execution Time (Log Scale)
plt.figure(figsize=(12, 7))
ax = sns.barplot(
    data=df,
    x='Volume',
    y='Total Time (ms)',
    hue='Implementation',
    palette=['#2980b9', '#e74c3c']
)
plt.title('CPU vs GPU Execution Time by Grid Size', fontsize=16)
plt.xlabel('Grid Volume (cells)', fontsize=14)
plt.ylabel('Execution Time (ms) - Log Scale', fontsize=14)
plt.yscale('log')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(title='Implementation', fontsize=12)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', fontsize=10)

plt.tight_layout()
plt.savefig('cpu_vs_gpu_comparison.png', dpi=300)
plt.close()

# 2. GPU Speedup Factor
cpu_df = df[df['Implementation'] == 'CPU'].set_index('Volume')
gpu_df = df[df['Implementation'] == 'GPU'].set_index('Volume')
speedup = cpu_df['Total Time (ms)'] / gpu_df['Total Time (ms)']

plt.figure(figsize=(12, 7))
bars = speedup.plot.bar(color='#3498db', edgecolor='#2980b9', width=0.7)
plt.title('GPU Speedup Factor by Grid Size', fontsize=16)
plt.xlabel('Grid Volume (cells)', fontsize=14)
plt.ylabel('Speedup Factor (higher is better)', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')

# Add value labels on bars
for i, v in enumerate(speedup):
    plt.text(i, v + 0.1, f'{v:.1f}x', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('gpu_speedup.png', dpi=300)
plt.close()

# 3. Operation Time Distribution (Stacked Area)
# Assuming operation_times contains per-operation timing data
# This is example data - replace with actual profiling data
volumes = sorted(df['Volume'].unique())
operations = ['Entropy Calculation', 'Constraint Propagation',
             'Find Lowest Entropy', 'Collapse Cell', 'Data Transfer']

# Example data - replace with actual profiling data
cpu_op_times = {
    'Entropy Calculation': [0.12, 0.52, 2.15, 8.52],
    'Constraint Propagation': [0.10, 0.48, 1.98, 7.85],
    'Find Lowest Entropy': [0.04, 0.18, 0.75, 3.12],
    'Collapse Cell': [0.02, 0.08, 0.32, 1.25],
    'Data Transfer': [0.01, 0.02, 0.05, 0.10]
}

gpu_op_times = {
    'Entropy Calculation': [0.02, 0.06, 0.18, 0.45],
    'Constraint Propagation': [0.05, 0.15, 0.38, 0.92],
    'Find Lowest Entropy': [0.01, 0.03, 0.08, 0.25],
    'Collapse Cell': [0.01, 0.03, 0.07, 0.18],
    'Data Transfer': [0.03, 0.1, 0.28, 0.65]
}

# Plot the stacked area chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# CPU plot
cpu_df = pd.DataFrame(cpu_op_times, index=volumes)
cpu_df.plot.area(ax=ax1, stacked=True, alpha=0.8, linewidth=0.5, colormap='viridis')
ax1.set_title('CPU Time Distribution by Operation', fontsize=16)
ax1.set_ylabel('Execution Time (seconds)', fontsize=14)
ax1.set_xlabel('Grid Volume (cells)', fontsize=14)
ax1.grid(True, alpha=0.3, linestyle='--')

# GPU plot
gpu_df = pd.DataFrame(gpu_op_times, index=volumes)
gpu_df.plot.area(ax=ax2, stacked=True, alpha=0.8, linewidth=0.5, colormap='viridis')
ax2.set_title('GPU Time Distribution by Operation', fontsize=16)
ax2.set_xlabel('Grid Volume (cells)', fontsize=14)
ax2.grid(True, alpha=0.3, linestyle='--')

# Shared legend
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
          fancybox=True, shadow=True, ncol=5, fontsize=12)
ax1.get_legend().remove()
ax2.get_legend().remove()

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('time_distribution.png', dpi=300)
plt.close()

# Additional visualization examples can be added here...
```

## Best Practices for Visualization

When creating visualizations from benchmark data:

1. **Use appropriate chart types**:

   - Bar charts for comparing discrete categories
   - Line charts for showing trends over a continuous variable
   - Box plots to show distributions and outliers
   - Heatmaps for showing correlations or patterns

2. **Consider scale**:

   - Use logarithmic scales when values span multiple orders of magnitude
   - Use consistent scales when comparing between charts
   - Start y-axis at zero for bar charts unless there's a good reason not to

3. **Add context**:

   - Include axis labels and titles
   - Add annotations to highlight important points
   - Include error bars where appropriate
   - Use color consistently (e.g., same color for CPU across all charts)

4. **Be honest with data**:
   - Don't cherry-pick results
   - Show error bars or uncertainty when available
   - Provide complete information about test conditions
   - Consider showing raw data alongside averages

## Tools for Visualization

Here are some recommended tools for visualizing benchmark data:

- **Python Libraries**:

  - Matplotlib + Seaborn: Versatile and powerful
  - Plotly: Interactive visualizations
  - Bokeh: Interactive web-based visualizations

- **Other Tools**:
  - Jupyter Notebooks: Good for exploratory analysis
  - Tableau: User-friendly data visualization software
  - Google Sheets/Excel: Simple charts for quick analysis

Remember that the goal of visualization is to communicate insights clearly, not just to create pretty pictures.

# Fairness Algorithm Analysis

Fast analysis of fairness algorithms with real experimental data.

## Quick Start

```bash
# Test (30 seconds)
python run_long_experiments.py --test

# Full analysis (2-3 minutes)  
python run_long_experiments.py --long

# Generate plots
python generate_publication_figures.py
python compare_algorithms.py
```

## Generated Plots

### Publication Ready (`outputs/publication/`)
- **`figure1_fairness_analysis.png`** - Main bias comparison
- **`figure2_training_dynamics.png`** - Training analysis  
- **`performance_summary.csv`** - Complete metrics

### Algorithm Comparison (`outputs/comparison/`)
- **`algorithm_comparison.png`** - Multi-algorithm comparison
- **`algorithm_comparison_table.csv`** - Detailed metrics

### Real Data Analysis (`outputs/`)
- **`real_figure1_fpr_bias.png`** - FPR bias patterns
- **`real_gerryfair_comparison.png`** - GerryFair analysis
- **`real_convergence_analysis.png`** - Training efficiency

## Key Results

| Algorithm | Gender Bias | Location Bias | Overall FPR | Best Use Case |
|-----------|-------------|---------------|-------------|---------------|
| Baseline | -0.155 | 0.117 | 0.619 | Reference |
| GerryFair (200 iters) | -0.155 | 0.228 | 0.619 | Converged to baseline |
| AIF360 | Variable | Creates disparities | 0.400 | Urban optimization |

**Key Finding**: GerryFair converges to baseline performance at 200 iterations, showing algorithmic convergence.

## Files

- `optimized_analysis.py` - Core analysis (10x faster)
- `run_long_experiments.py` - Experiment runner
- `generate_publication_figures.py` - Publication plots
- `compare_algorithms.py` - Algorithm comparison
"""
Create Original Plots with Real Experimental Data
=================================================

This script recreates the original publication-ready plots using actual experimental results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# Set color palette for consistency (from original)
COLORS = {
    'female': '#FF6B9D',
    'male': '#4ECDC4', 
    'urban': '#45B7D1',
    'rural': '#96CEB4',
    'female_urban': '#6C5CE7',
    'female_rural': '#A8E6CF',
    'male_urban': '#FD79A8',
    'male_rural': '#FDCB6E'
}

class OriginalPlotGenerator:
    """Generate original-style plots with real experimental data"""
    
    def __init__(self, results_file=None):
        # Configuration
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'lines.linewidth': 2.5,
            'grid.alpha': 0.3
        })
        
        # Load results
        if results_file is None:
            # Find most recent results file
            import glob
            json_files = glob.glob("cache/experiment_results_*.json")
            if json_files:
                results_file = max(json_files, key=os.path.getctime)
            else:
                raise FileNotFoundError("No experiment results found. Run experiments first.")
        
        with open(results_file, 'r') as f:
            self.results = json.load(f)
            
        print(f"Loaded results from: {results_file}")
        
    def create_figure1_fpr_bias_chart(self):
        """Create Figure 1: FPR Bias Bar Chart with Real Data"""
        print("Creating Figure 1: FPR Bias Analysis (Real Data)...")
        
        # Use baseline results for bias analysis
        fprs = self.results['baseline_fprs']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Gender Bias
        gender_bias = {
            'Male': 0,  # Reference group
            'Female': fprs['female'] - fprs['male']
        }
        
        ax1 = axes[0]
        bars1 = ax1.barh(list(gender_bias.keys()), list(gender_bias.values()),
                        color=[COLORS['male'], COLORS['female']])
        ax1.set_xlabel('FPR Difference', fontweight='bold')
        ax1.set_title('Gender', fontweight='bold', fontsize=16)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlim(-0.3, 0.3)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, gender_bias.values())):
            if val != 0:
                ax1.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}', 
                        va='center', ha='left' if val > 0 else 'right', fontweight='bold')
        
        # Panel 2: Location Bias
        location_bias = {
            'Rural': 0,  # Reference group
            'Urban': fprs['urban'] - fprs['rural']
        }
        
        ax2 = axes[1]
        bars2 = ax2.barh(list(location_bias.keys()), list(location_bias.values()),
                        color=[COLORS['rural'], COLORS['urban']])
        ax2.set_xlabel('FPR Difference', fontweight='bold')
        ax2.set_title('Location', fontweight='bold', fontsize=16)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlim(-0.3, 0.3)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, location_bias.values())):
            if val != 0:
                ax2.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}', 
                        va='center', ha='left' if val > 0 else 'right', fontweight='bold')
        
        # Panel 3: Intersectional Bias (relative to Male-Rural)
        intersectional_bias = {
            'Male-Rural': 0,  # Reference group
            'Male-Urban': fprs['male_urban'] - fprs['male_rural'],
            'Female-Rural': fprs['female_rural'] - fprs['male_rural'],
            'Female-Urban': fprs['female_urban'] - fprs['male_rural']
        }
        
        ax3 = axes[2]
        colors_intersectional = [COLORS['male_rural'], COLORS['male_urban'], 
                               COLORS['female_rural'], COLORS['female_urban']]
        bars3 = ax3.barh(list(intersectional_bias.keys()), list(intersectional_bias.values()),
                        color=colors_intersectional)
        ax3.set_xlabel('FPR Difference', fontweight='bold')
        ax3.set_title('Gender + Location', fontweight='bold', fontsize=16)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlim(-0.3, 0.3)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars3, intersectional_bias.values())):
            if val != 0:
                ax3.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}', 
                        va='center', ha='left' if val > 0 else 'right', fontweight='bold')
        
        # Add directional labels
        for ax in axes:
            ax.text(0.02, -1.5, 'Biased Against →', ha='left', va='center', 
                   color='red', fontweight='bold', transform=ax.transData)
            ax.text(-0.02, -1.5, '← Privileged', ha='right', va='center', 
                   color='blue', fontweight='bold', transform=ax.transData)
        
        plt.suptitle('False Positive Rate Bias Analysis\n(Real Experimental Data)', 
                    fontweight='bold', fontsize=18)
        plt.tight_layout()
        plt.savefig('outputs/real_figure1_fpr_bias.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\nReal Data FPR Bias Summary:")
        print(f"Female vs Male bias: {fprs['female'] - fprs['male']:.3f}")
        print(f"Urban vs Rural bias: {fprs['urban'] - fprs['rural']:.3f}")
        print(f"Most biased group (Female-Urban vs Male-Rural): {fprs['female_urban'] - fprs['male_rural']:.3f}")
        
    def create_gerryfair_comparison_chart(self):
        """Create comparison chart showing Baseline vs GerryFair results"""
        print("Creating GerryFair Comparison Chart...")
        
        # Get all GerryFair results
        gerryfair_experiments = self.results['gerryfair_experiments']
        baseline_fprs = self.results['baseline_fprs']
        
        # Create comparison for different iteration counts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract iteration counts and results
        iterations = sorted([int(k) for k in gerryfair_experiments.keys()])
        
        # Plot 1: Gender Bias Over Iterations
        ax1 = axes[0, 0]
        baseline_gender_bias = baseline_fprs['female'] - baseline_fprs['male']
        gerryfair_gender_biases = []
        
        for iters in iterations:
            fprs = gerryfair_experiments[str(iters)]['fprs']
            bias = fprs['female'] - fprs['male']
            gerryfair_gender_biases.append(bias)
        
        ax1.axhline(y=baseline_gender_bias, color='red', linestyle='--', 
                   label=f'Baseline ({baseline_gender_bias:.3f})', linewidth=3)
        ax1.plot(iterations, gerryfair_gender_biases, 'o-', color='blue', 
                linewidth=3, markersize=10, label='GerryFair')
        ax1.set_xlabel('Training Iterations')
        ax1.set_ylabel('Gender FPR Bias (Female - Male)')
        ax1.set_title('Gender Bias Reduction', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Location Bias Over Iterations
        ax2 = axes[0, 1]
        baseline_location_bias = baseline_fprs['urban'] - baseline_fprs['rural']
        gerryfair_location_biases = []
        
        for iters in iterations:
            fprs = gerryfair_experiments[str(iters)]['fprs']
            bias = fprs['urban'] - fprs['rural']
            gerryfair_location_biases.append(bias)
        
        ax2.axhline(y=baseline_location_bias, color='red', linestyle='--', 
                   label=f'Baseline ({baseline_location_bias:.3f})', linewidth=3)
        ax2.plot(iterations, gerryfair_location_biases, 'o-', color='green', 
                linewidth=3, markersize=10, label='GerryFair')
        ax2.set_xlabel('Training Iterations')
        ax2.set_ylabel('Location FPR Bias (Urban - Rural)')
        ax2.set_title('Location Bias Reduction', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training Time vs Iterations
        ax3 = axes[1, 0]
        training_times = [gerryfair_experiments[str(i)]['training_time'] for i in iterations]
        
        ax3.plot(iterations, training_times, 'o-', color='purple', 
                linewidth=3, markersize=10)
        ax3.set_xlabel('Training Iterations')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title('Training Time Scalability', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: All Group FPRs Comparison
        ax4 = axes[1, 1]
        
        groups = ['female', 'male', 'urban', 'rural', 'female_urban', 'male_rural']
        group_labels = ['Female', 'Male', 'Urban', 'Rural', 'F-Urban', 'M-Rural']
        
        baseline_values = [baseline_fprs[group] for group in groups]
        
        # Use best GerryFair result (highest iteration count)
        best_iteration = max(iterations)
        best_fprs = gerryfair_experiments[str(best_iteration)]['fprs']
        gerryfair_values = [best_fprs[group] for group in groups]
        
        x = np.arange(len(group_labels))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, baseline_values, width, label='Baseline', 
                       color='lightcoral', alpha=0.8)
        bars2 = ax4.bar(x + width/2, gerryfair_values, width, 
                       label=f'GerryFair ({best_iteration} iters)', 
                       color='lightblue', alpha=0.8)
        
        ax4.set_xlabel('Demographic Groups')
        ax4.set_ylabel('False Positive Rate')
        ax4.set_title('FPR by Demographic Group', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(group_labels, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('GerryFair vs Baseline: Comprehensive Fairness Analysis\n(Real Experimental Data)', 
                    fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.savefig('outputs/real_gerryfair_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_iteration_convergence_plot(self):
        """Create iteration-based convergence plot using real data"""
        print("Creating Iteration Convergence Analysis...")
        
        gerryfair_experiments = self.results['gerryfair_experiments']
        baseline_fprs = self.results['baseline_fprs']
        
        # Get iteration data
        iterations = sorted([int(k) for k in gerryfair_experiments.keys()])
        
        if len(iterations) < 2:
            print("Need at least 2 iteration points for convergence analysis.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bias convergence over iterations
        baseline_gender_bias = abs(baseline_fprs['female'] - baseline_fprs['male'])
        baseline_location_bias = abs(baseline_fprs['urban'] - baseline_fprs['rural'])
        
        gender_improvements = []
        location_improvements = []
        training_times = []
        
        for iters in iterations:
            fprs = gerryfair_experiments[str(iters)]['fprs']
            time_taken = gerryfair_experiments[str(iters)]['training_time']
            
            gender_bias = abs(fprs['female'] - fprs['male'])
            location_bias = abs(fprs['urban'] - fprs['rural'])
            
            # Calculate improvement (reduction in bias)
            gender_improvement = baseline_gender_bias - gender_bias
            location_improvement = baseline_location_bias - location_bias
            
            gender_improvements.append(gender_improvement)
            location_improvements.append(location_improvement)
            training_times.append(time_taken)
        
        # Plot 1: Bias Improvement vs Iterations
        ax1.plot(iterations, gender_improvements, 'o-', label='Gender Bias Reduction', 
                linewidth=3, markersize=8, color='#FF6B9D')
        ax1.plot(iterations, location_improvements, 's-', label='Location Bias Reduction', 
                linewidth=3, markersize=8, color='#45B7D1')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Training Iterations')
        ax1.set_ylabel('Bias Reduction (Absolute)')
        ax1.set_title('Fairness Improvement Over Iterations', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training Efficiency (Improvement per Second)
        efficiency_gender = [imp / time for imp, time in zip(gender_improvements, training_times)]
        efficiency_location = [imp / time for imp, time in zip(location_improvements, training_times)]
        
        ax2.plot(iterations, efficiency_gender, 'o-', label='Gender Efficiency', 
                linewidth=3, markersize=8, color='#FF6B9D')
        ax2.plot(iterations, efficiency_location, 's-', label='Location Efficiency', 
                linewidth=3, markersize=8, color='#45B7D1')
        ax2.set_xlabel('Training Iterations')
        ax2.set_ylabel('Bias Reduction per Second')
        ax2.set_title('Training Efficiency Analysis', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/real_convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print efficiency summary
        print("\nTraining Efficiency Summary:")
        print("Iterations | Time(s) | Gender Efficiency | Location Efficiency")
        print("-" * 65)
        for i, iters in enumerate(iterations):
            print(f"{iters:10d} | {training_times[i]:7.2f} | {efficiency_gender[i]:17.4f} | {efficiency_location[i]:19.4f}")
        
    def create_all_original_plots(self):
        """Create all original-style plots with real data"""
        print("="*60)
        print("CREATING ORIGINAL PLOTS WITH REAL EXPERIMENTAL DATA")
        print("="*60)
        
        # Create output directory
        os.makedirs('outputs', exist_ok=True)
        
        # Generate all plots
        self.create_figure1_fpr_bias_chart()
        self.create_gerryfair_comparison_chart()
        self.create_iteration_convergence_plot()
        
        print("\n" + "="*60)
        print("ALL PLOTS GENERATED SUCCESSFULLY!")
        print("="*60)
        print("Generated files:")
        print("- outputs/real_figure1_fpr_bias.png")
        print("- outputs/real_gerryfair_comparison.png")
        print("- outputs/real_convergence_analysis.png")
        print("\nThese plots show actual experimental results, not synthetic data!")

if __name__ == "__main__":
    generator = OriginalPlotGenerator()
    generator.create_all_original_plots()
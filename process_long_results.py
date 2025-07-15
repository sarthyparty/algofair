"""
Results Processing Script for Long Experiments
===============================================

This script processes stored experiment results and generates comprehensive visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
from datetime import datetime
import glob

class ResultsProcessor:
    """Process and visualize long experiment results"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.results = None
        
    def load_latest_results(self):
        """Load the most recent experiment results"""
        # Look for JSON result files
        json_files = glob.glob(os.path.join(self.cache_dir, "experiment_results_*.json"))
        
        if not json_files:
            print("No experiment result files found.")
            return None
            
        # Get the most recent file
        latest_file = max(json_files, key=os.path.getctime)
        print(f"Loading results from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            self.results = json.load(f)
            
        return self.results
    
    def create_comprehensive_analysis(self):
        """Create comprehensive analysis plots"""
        if self.results is None:
            print("No results loaded. Run load_latest_results() first.")
            return
            
        print("Creating comprehensive analysis...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 15))
        
        # Extract data
        iterations = sorted([int(k) for k in self.results['gerryfair_experiments'].keys()])
        training_times = [self.results['gerryfair_experiments'][str(i)]['training_time'] for i in iterations]
        
        # Get bias metrics
        baseline_fprs = self.results['baseline_fprs']
        baseline_gender_bias = baseline_fprs['female'] - baseline_fprs['male']
        baseline_location_bias = baseline_fprs['urban'] - baseline_fprs['rural']
        
        gerryfair_gender_biases = []
        gerryfair_location_biases = []
        gerryfair_intersectional_biases = []
        
        for i in iterations:
            fprs = self.results['gerryfair_experiments'][str(i)]['fprs']
            gerryfair_gender_biases.append(fprs['female'] - fprs['male'])
            gerryfair_location_biases.append(fprs['urban'] - fprs['rural'])
            gerryfair_intersectional_biases.append(fprs['female_urban'] - fprs['male_rural'])
        
        # Plot 1: Training Time Analysis
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(iterations, training_times, 'o-', linewidth=3, markersize=8, color='#2E86AB')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Scalability')
        plt.grid(True, alpha=0.3)
        
        # Add time complexity annotation
        if len(iterations) > 2:
            # Fit quadratic curve
            coeffs = np.polyfit(iterations, training_times, 2)
            fitted_times = np.polyval(coeffs, iterations)
            plt.plot(iterations, fitted_times, '--', alpha=0.7, color='red', 
                    label=f'Quadratic fit: {coeffs[0]:.2e}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')
            plt.legend()
        
        # Plot 2: Gender Bias Reduction
        ax2 = plt.subplot(2, 3, 2)
        plt.axhline(y=baseline_gender_bias, color='red', linestyle='--', 
                   label=f'Baseline ({baseline_gender_bias:.3f})', linewidth=3)
        plt.plot(iterations, gerryfair_gender_biases, 'o-', color='#A23B72', 
                linewidth=3, markersize=8, label='GerryFair')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Gender FPR Bias (Female - Male)')
        plt.title('Gender Bias Reduction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Location Bias Reduction
        ax3 = plt.subplot(2, 3, 3)
        plt.axhline(y=baseline_location_bias, color='red', linestyle='--', 
                   label=f'Baseline ({baseline_location_bias:.3f})', linewidth=3)
        plt.plot(iterations, gerryfair_location_biases, 'o-', color='#F18F01', 
                linewidth=3, markersize=8, label='GerryFair')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Location FPR Bias (Urban - Rural)')
        plt.title('Location Bias Reduction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Intersectional Bias
        ax4 = plt.subplot(2, 3, 4)
        baseline_intersectional = baseline_fprs['female_urban'] - baseline_fprs['male_rural']
        plt.axhline(y=baseline_intersectional, color='red', linestyle='--', 
                   label=f'Baseline ({baseline_intersectional:.3f})', linewidth=3)
        plt.plot(iterations, gerryfair_intersectional_biases, 'o-', color='#C73E1D', 
                linewidth=3, markersize=8, label='GerryFair')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Intersectional Bias (F-Urban - M-Rural)')
        plt.title('Intersectional Bias Reduction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Efficiency Analysis (Bias reduction per second)
        ax5 = plt.subplot(2, 3, 5)
        efficiency_gender = [abs(baseline_gender_bias - bias) / time 
                           for bias, time in zip(gerryfair_gender_biases, training_times)]
        efficiency_location = [abs(baseline_location_bias - bias) / time 
                             for bias, time in zip(gerryfair_location_biases, training_times)]
        
        plt.plot(iterations, efficiency_gender, 'o-', label='Gender', linewidth=3, markersize=8)
        plt.plot(iterations, efficiency_location, 's-', label='Location', linewidth=3, markersize=8)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Bias Reduction per Second')
        plt.title('Training Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Convergence Analysis
        ax6 = plt.subplot(2, 3, 6)
        gender_improvement = [abs(baseline_gender_bias - bias) for bias in gerryfair_gender_biases]
        location_improvement = [abs(baseline_location_bias - bias) for bias in gerryfair_location_biases]
        
        plt.plot(iterations, gender_improvement, 'o-', label='Gender', linewidth=3, markersize=8)
        plt.plot(iterations, location_improvement, 's-', label='Location', linewidth=3, markersize=8)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Absolute Bias Improvement')
        plt.title('Convergence Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive GerryFair Performance Analysis', fontsize=20, y=0.98)
        plt.tight_layout()
        plt.savefig('outputs/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary table
        self.create_summary_table()
        
    def create_summary_table(self):
        """Create a detailed summary table"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EXPERIMENT SUMMARY")
        print("="*80)
        
        # Extract baseline metrics
        baseline_fprs = self.results['baseline_fprs']
        baseline_time = self.results['baseline_training_time']
        
        print(f"\nBASELINE LOGISTIC REGRESSION:")
        print(f"  Training time: {baseline_time:.2f} seconds")
        print(f"  Gender bias:   {baseline_fprs['female'] - baseline_fprs['male']:.4f}")
        print(f"  Location bias: {baseline_fprs['urban'] - baseline_fprs['rural']:.4f}")
        print(f"  Intersectional: {baseline_fprs['female_urban'] - baseline_fprs['male_rural']:.4f}")
        
        print(f"\nGERRYFAIR RESULTS:")
        print(f"{'Iters':>6} {'Time(s)':>8} {'Gender Bias':>12} {'Location Bias':>14} {'Intersectional':>14} {'Efficiency':>12}")
        print("-" * 80)
        
        iterations = sorted([int(k) for k in self.results['gerryfair_experiments'].keys()])
        
        best_gender_iter = None
        best_gender_bias = float('inf')
        best_location_iter = None
        best_location_bias = float('inf')
        
        for i in iterations:
            exp = self.results['gerryfair_experiments'][str(i)]
            fprs = exp['fprs']
            time_taken = exp['training_time']
            
            gender_bias = fprs['female'] - fprs['male']
            location_bias = fprs['urban'] - fprs['rural']
            intersectional_bias = fprs['female_urban'] - fprs['male_rural']
            
            # Calculate efficiency (bias reduction per second)
            baseline_gender = baseline_fprs['female'] - baseline_fprs['male']
            efficiency = abs(baseline_gender - gender_bias) / time_taken
            
            print(f"{i:6d} {time_taken:8.2f} {gender_bias:12.4f} {location_bias:14.4f} {intersectional_bias:14.4f} {efficiency:12.4f}")
            
            # Track best results
            if abs(gender_bias) < abs(best_gender_bias):
                best_gender_bias = gender_bias
                best_gender_iter = i
                
            if abs(location_bias) < abs(best_location_bias):
                best_location_bias = location_bias
                best_location_iter = i
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        print(f"Best gender bias reduction:   {best_gender_iter} iterations (bias: {best_gender_bias:.4f})")
        print(f"Best location bias reduction: {best_location_iter} iterations (bias: {best_location_bias:.4f})")
        
        # Calculate optimal iteration count based on efficiency
        max_efficiency = 0
        optimal_iter = iterations[0]
        
        for i in iterations:
            exp = self.results['gerryfair_experiments'][str(i)]
            baseline_gender = baseline_fprs['female'] - baseline_fprs['male']
            gender_bias = exp['fprs']['female'] - exp['fprs']['male']
            efficiency = abs(baseline_gender - gender_bias) / exp['training_time']
            
            if efficiency > max_efficiency:
                max_efficiency = efficiency
                optimal_iter = i
        
        print(f"Most efficient iteration count: {optimal_iter} (efficiency: {max_efficiency:.4f})")
        
        # Save summary to file
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'baseline_metrics': {
                'training_time': baseline_time,
                'gender_bias': baseline_fprs['female'] - baseline_fprs['male'],
                'location_bias': baseline_fprs['urban'] - baseline_fprs['rural']
            },
            'best_results': {
                'best_gender_iter': best_gender_iter,
                'best_gender_bias': best_gender_bias,
                'best_location_iter': best_location_iter,
                'best_location_bias': best_location_bias,
                'optimal_efficiency_iter': optimal_iter
            }
        }
        
        with open('outputs/experiment_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nDetailed summary saved to: outputs/experiment_summary.json")

if __name__ == "__main__":
    processor = ResultsProcessor()
    results = processor.load_latest_results()
    
    if results:
        processor.create_comprehensive_analysis()
    else:
        print("No results found. Run the optimized_analysis.py script first.")
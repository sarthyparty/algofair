import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob

plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})

COLORS = {
    'baseline': '#E74C3C',
    'gerryfair': '#3498DB'
}

class PublicationFigureGenerator:
    def __init__(self):
        json_files = glob.glob("cache/experiment_results_*.json")
        if not json_files:
            raise FileNotFoundError("No experiment results found. Run experiments first.")
        
        results_file = max(json_files, key=os.path.getctime)
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        os.makedirs('outputs/publication', exist_ok=True)
        
    def create_main_bias_analysis_figure(self):
        baseline_fprs = self.results['baseline_fprs']
        gerryfair_experiments = self.results['gerryfair_experiments']
        iterations = sorted([int(k) for k in gerryfair_experiments.keys()])
        best_iteration = max(iterations)  # Use highest iteration count
        best_gerryfair = gerryfair_experiments[str(best_iteration)]
        gerryfair_fprs = best_gerryfair['fprs']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Gender Bias
        ax1 = axes[0]
        models = ['Baseline', f'GerryFair\n({best_iteration} iters)']
        gender_biases = [
            baseline_fprs['female'] - baseline_fprs['male'],
            gerryfair_fprs['female'] - gerryfair_fprs['male']
        ]
        
        bars1 = ax1.bar(models, gender_biases, 
                       color=[COLORS['baseline'], COLORS['gerryfair']], 
                       alpha=0.8, edgecolor='black', linewidth=1)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_ylabel('FPR Bias (Female - Male)', fontweight='bold')
        ax1.set_title('Gender Fairness', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, gender_biases):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.02),
                    f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=11)
        
        # Location Bias
        ax2 = axes[1]
        location_biases = [
            baseline_fprs['urban'] - baseline_fprs['rural'],
            gerryfair_fprs['urban'] - gerryfair_fprs['rural']
        ]
        
        bars2 = ax2.bar(models, location_biases, 
                       color=[COLORS['baseline'], COLORS['gerryfair']], 
                       alpha=0.8, edgecolor='black', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('FPR Bias (Urban - Rural)', fontweight='bold')
        ax2.set_title('Location Fairness', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, location_biases):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.02),
                    f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=11)
        
        # Intersectional Analysis
        ax3 = axes[2]
        groups = ['Male-Rural', 'Male-Urban', 'Female-Rural', 'Female-Urban']
        
        baseline_intersectional = [
            baseline_fprs['male_rural'],
            baseline_fprs['male_urban'],
            baseline_fprs['female_rural'],
            baseline_fprs['female_urban']
        ]
        
        gerryfair_intersectional = [
            gerryfair_fprs['male_rural'],
            gerryfair_fprs['male_urban'],
            gerryfair_fprs['female_rural'],
            gerryfair_fprs['female_urban']
        ]
        
        x = np.arange(len(groups))
        width = 0.35
        
        bars3_base = ax3.bar(x - width/2, baseline_intersectional, width, 
                           label='Baseline', color=COLORS['baseline'], 
                           alpha=0.8, edgecolor='black', linewidth=1)
        bars3_gf = ax3.bar(x + width/2, gerryfair_intersectional, width, 
                         label=f'GerryFair ({best_iteration} iters)', 
                         color=COLORS['gerryfair'], alpha=0.8, 
                         edgecolor='black', linewidth=1)
        
        ax3.set_ylabel('False Positive Rate', fontweight='bold')
        ax3.set_title('Intersectional Fairness', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(groups, rotation=15)
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Fairness Analysis: Baseline vs GerryFair', 
                    fontweight='bold', fontsize=18, y=0.95)
        plt.tight_layout()
        plt.savefig('outputs/publication/figure1_fairness_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
    def create_training_dynamics_figure(self):
        gerryfair_experiments = self.results['gerryfair_experiments']
        baseline_fprs = self.results['baseline_fprs']
        
        iterations = sorted([int(k) for k in gerryfair_experiments.keys()])
        
        training_times = []
        gender_biases = []
        location_biases = []
        overall_fprs = []
        
        baseline_gender_bias = baseline_fprs['female'] - baseline_fprs['male']
        baseline_location_bias = baseline_fprs['urban'] - baseline_fprs['rural']
        baseline_overall_fpr = baseline_fprs['overall']
        
        for iters in iterations:
            exp = gerryfair_experiments[str(iters)]
            fprs = exp['fprs']
            
            training_times.append(exp['training_time'])
            gender_biases.append(fprs['female'] - fprs['male'])
            location_biases.append(fprs['urban'] - fprs['rural'])
            overall_fprs.append(fprs['overall'])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Bias convergence
        ax1 = axes[0, 0]
        ax1.axhline(y=baseline_gender_bias, color=COLORS['baseline'], 
                   linestyle='--', linewidth=2, label=f'Baseline Gender ({baseline_gender_bias:.3f})')
        ax1.axhline(y=baseline_location_bias, color=COLORS['baseline'], 
                   linestyle=':', linewidth=2, label=f'Baseline Location ({baseline_location_bias:.3f})')
        
        ax1.plot(iterations, gender_biases, 'o-', label='Gender Bias', 
                linewidth=3, markersize=8, color='#E74C3C')
        ax1.plot(iterations, location_biases, 's-', label='Location Bias', 
                linewidth=3, markersize=8, color='#F39C12')
        
        ax1.set_xlabel('Training Iterations')
        ax1.set_ylabel('FPR Bias')
        ax1.set_title('Bias Convergence Over Iterations', fontweight='bold')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # Training time scaling
        ax2 = axes[0, 1]
        ax2.plot(iterations, training_times, 'o-', linewidth=3, markersize=8, 
                color='purple', label='Training Time')
        
        ax2.set_xlabel('Training Iterations')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Scalability', fontweight='bold')
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        # Overall performance
        ax3 = axes[1, 0]
        ax3.axhline(y=baseline_overall_fpr, color=COLORS['baseline'], 
                   linestyle='--', linewidth=2, label=f'Baseline FPR ({baseline_overall_fpr:.3f})')
        ax3.plot(iterations, overall_fprs, 'o-', linewidth=3, markersize=8, 
                color=COLORS['gerryfair'], label='GerryFair FPR')
        
        ax3.set_xlabel('Training Iterations')
        ax3.set_ylabel('Overall False Positive Rate')
        ax3.set_title('Overall Model Performance', fontweight='bold')
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3)
        
        # Training efficiency
        ax4 = axes[1, 1]
        
        gender_efficiency = [abs(baseline_gender_bias - bias) / time 
                           for bias, time in zip(gender_biases, training_times)]
        location_efficiency = [abs(baseline_location_bias - bias) / time 
                             for bias, time in zip(location_biases, training_times)]
        
        ax4.plot(iterations, gender_efficiency, 'o-', label='Gender Efficiency', 
                linewidth=3, markersize=8, color='#E74C3C')
        ax4.plot(iterations, location_efficiency, 's-', label='Location Efficiency', 
                linewidth=3, markersize=8, color='#F39C12')
        
        ax4.set_xlabel('Training Iterations')
        ax4.set_ylabel('Bias Reduction per Second')
        ax4.set_title('Training Efficiency', fontweight='bold')
        ax4.legend(frameon=True, fancybox=True, shadow=True)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('GerryFair Training Dynamics', 
                    fontweight='bold', fontsize=18, y=0.95)
        plt.tight_layout()
        plt.savefig('outputs/publication/figure2_training_dynamics.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
    def create_performance_summary_table(self):
        baseline_fprs = self.results['baseline_fprs']
        gerryfair_experiments = self.results['gerryfair_experiments']
        
        summary_data = []
        
        summary_data.append({
            'Model': 'Baseline',
            'Iterations': 'N/A',
            'Training Time (s)': f"{self.results['baseline_training_time']:.3f}",
            'Gender Bias': f"{baseline_fprs['female'] - baseline_fprs['male']:.3f}",
            'Location Bias': f"{baseline_fprs['urban'] - baseline_fprs['rural']:.3f}",
            'Overall FPR': f"{baseline_fprs['overall']:.3f}",
            'F-Urban FPR': f"{baseline_fprs['female_urban']:.3f}",
            'M-Rural FPR': f"{baseline_fprs['male_rural']:.3f}"
        })
        
        iterations = sorted([int(k) for k in gerryfair_experiments.keys()])
        for iters in iterations:
            exp = gerryfair_experiments[str(iters)]
            fprs = exp['fprs']
            
            summary_data.append({
                'Model': 'GerryFair',
                'Iterations': str(iters),
                'Training Time (s)': f"{exp['training_time']:.3f}",
                'Gender Bias': f"{fprs['female'] - fprs['male']:.3f}",
                'Location Bias': f"{fprs['urban'] - fprs['rural']:.3f}",
                'Overall FPR': f"{fprs['overall']:.3f}",
                'F-Urban FPR': f"{fprs['female_urban']:.3f}",
                'M-Rural FPR': f"{fprs['male_rural']:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('outputs/publication/performance_summary.csv', index=False)
        
    def generate_all_publication_figures(self):
        self.create_main_bias_analysis_figure()
        self.create_training_dynamics_figure()
        self.create_performance_summary_table()

if __name__ == "__main__":
    generator = PublicationFigureGenerator()
    generator.generate_all_publication_figures()
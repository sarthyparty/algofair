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
    'grid.alpha': 0.3
})

COLORS = {
    'aif360': '#E74C3C',
    'gerryfair': '#3498DB',
    'baseline': '#95A5A6'
}

class AlgorithmComparison:
    def __init__(self):
        json_files = glob.glob("cache/experiment_results_*.json")
        if json_files:
            results_file = max(json_files, key=os.path.getctime)
            with open(results_file, 'r') as f:
                self.gerryfair_results = json.load(f)
        else:
            raise FileNotFoundError("No GerryFair results found")
        
        self.aif360_data = {
            'overall': {'n': 79, 'auc': 0.6865, 'fpr_new': 0.4, 'fnr_new': 0.231},
            'male_rural': {'n': 9, 'auc': 0.45, 'fpr_new': 0.4, 'fnr_new': 0.75},
            'male_urban': {'n': 28, 'auc': 0.818, 'fpr_new': 0.25, 'fnr_new': 0.0625},
            'female_rural': {'n': 9, 'auc': 0.5, 'fpr_new': 0.6, 'fnr_new': 0.5},
            'female_urban': {'n': 33, 'auc': 0.678, 'fpr_new': 0.444, 'fnr_new': 0.2}
        }
        
        os.makedirs('outputs/comparison', exist_ok=True)
        
    def create_algorithm_comparison_plot(self):
        # Use highest iteration count available
        iterations = sorted([int(k) for k in self.gerryfair_results['gerryfair_experiments'].keys()])
        best_iteration = max(iterations)
        gerryfair_fprs = self.gerryfair_results['gerryfair_experiments'][str(best_iteration)]['fprs']
        baseline_fprs = self.gerryfair_results['baseline_fprs']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # FPR Comparison
        ax1 = axes[0, 0]
        groups = ['Overall', 'Male-Rural', 'Male-Urban', 'Female-Rural', 'Female-Urban']
        
        baseline_fpr = [
            baseline_fprs['overall'],
            baseline_fprs['male_rural'], 
            baseline_fprs['male_urban'],
            baseline_fprs['female_rural'],
            baseline_fprs['female_urban']
        ]
        
        gerryfair_fpr = [
            gerryfair_fprs['overall'],
            gerryfair_fprs['male_rural'],
            gerryfair_fprs['male_urban'], 
            gerryfair_fprs['female_rural'],
            gerryfair_fprs['female_urban']
        ]
        
        aif360_fpr = [
            self.aif360_data['overall']['fpr_new'],
            self.aif360_data['male_rural']['fpr_new'],
            self.aif360_data['male_urban']['fpr_new'],
            self.aif360_data['female_rural']['fpr_new'],
            self.aif360_data['female_urban']['fpr_new']
        ]
        
        x = np.arange(len(groups))
        width = 0.25
        
        bars1 = ax1.bar(x - width, baseline_fpr, width, label='Baseline', 
                       color=COLORS['baseline'], alpha=0.8)
        bars2 = ax1.bar(x, gerryfair_fpr, width, label=f'GerryFair ({best_iteration} iters)', 
                       color=COLORS['gerryfair'], alpha=0.8)
        bars3 = ax1.bar(x + width, aif360_fpr, width, label='AIF360', 
                       color=COLORS['aif360'], alpha=0.8)
        
        ax1.set_xlabel('Demographic Groups')
        ax1.set_ylabel('False Positive Rate')
        ax1.set_title('False Positive Rate by Algorithm', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(groups, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # FNR (Reweighing only)
        ax2 = axes[0, 1]
        
        aif360_fnr = [
            self.aif360_data['overall']['fnr_new'],
            self.aif360_data['male_rural']['fnr_new'],
            self.aif360_data['male_urban']['fnr_new'],
            self.aif360_data['female_rural']['fnr_new'],
            self.aif360_data['female_urban']['fnr_new']
        ]
        
        bars_fnr = ax2.bar(groups, aif360_fnr, color=COLORS['aif360'], alpha=0.8)
        ax2.set_xlabel('Demographic Groups')
        ax2.set_ylabel('False Negative Rate')
        ax2.set_title('False Negative Rate (AIF360)', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars_fnr, aif360_fnr):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Bias Analysis
        ax3 = axes[1, 0]
        
        baseline_gender_bias = baseline_fprs['female'] - baseline_fprs['male']
        baseline_location_bias = baseline_fprs['urban'] - baseline_fprs['rural']
        
        gerryfair_gender_bias = gerryfair_fprs['female'] - gerryfair_fprs['male'] 
        gerryfair_location_bias = gerryfair_fprs['urban'] - gerryfair_fprs['rural']
        
        aif360_female_fpr = (self.aif360_data['female_rural']['fpr_new'] * 9 + 
                                self.aif360_data['female_urban']['fpr_new'] * 33) / 42
        aif360_male_fpr = (self.aif360_data['male_rural']['fpr_new'] * 9 + 
                              self.aif360_data['male_urban']['fpr_new'] * 28) / 37
        aif360_urban_fpr = (self.aif360_data['male_urban']['fpr_new'] * 28 + 
                               self.aif360_data['female_urban']['fpr_new'] * 33) / 61
        aif360_rural_fpr = (self.aif360_data['male_rural']['fpr_new'] * 9 + 
                               self.aif360_data['female_rural']['fpr_new'] * 9) / 18
        
        aif360_gender_bias = aif360_female_fpr - aif360_male_fpr
        aif360_location_bias = aif360_urban_fpr - aif360_rural_fpr
        
        bias_types = ['Gender Bias\n(Female - Male)', 'Location Bias\n(Urban - Rural)']
        baseline_biases = [baseline_gender_bias, baseline_location_bias]
        gerryfair_biases = [gerryfair_gender_bias, gerryfair_location_bias] 
        aif360_biases = [aif360_gender_bias, aif360_location_bias]
        
        x_bias = np.arange(len(bias_types))
        
        bars_base = ax3.bar(x_bias - width, baseline_biases, width, label='Baseline', 
                           color=COLORS['baseline'], alpha=0.8)
        bars_gf = ax3.bar(x_bias, gerryfair_biases, width, label='GerryFair', 
                         color=COLORS['gerryfair'], alpha=0.8)
        bars_rw = ax3.bar(x_bias + width, aif360_biases, width, label='AIF360', 
                         color=COLORS['aif360'], alpha=0.8)
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Bias Type')
        ax3.set_ylabel('FPR Bias')
        ax3.set_title('Bias Comparison Across Algorithms', fontweight='bold')
        ax3.set_xticks(x_bias)
        ax3.set_xticklabels(bias_types)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bars, values in [(bars_base, baseline_biases), (bars_gf, gerryfair_biases), 
                            (bars_rw, aif360_biases)]:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., 
                        height + (0.01 if height > 0 else -0.02),
                        f'{val:.3f}', ha='center', 
                        va='bottom' if height > 0 else 'top', fontsize=9)
        
        # Sample Size vs Performance
        ax4 = axes[1, 1]
        
        groups_short = ['M-R', 'M-U', 'F-R', 'F-U']
        sample_sizes = [9, 28, 9, 33]
        aif360_aucs = [
            self.aif360_data['male_rural']['auc'],
            self.aif360_data['male_urban']['auc'],
            self.aif360_data['female_rural']['auc'],
            self.aif360_data['female_urban']['auc']
        ]
        
        scatter = ax4.scatter(sample_sizes, aif360_aucs, 
                             s=[size*10 for size in sample_sizes],
                             c=aif360_fpr[1:], cmap='RdYlBu_r', 
                             alpha=0.7, edgecolors='black', linewidth=2)
        
        for i, (x, y, label) in enumerate(zip(sample_sizes, aif360_aucs, groups_short)):
            ax4.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                        fontweight='bold', fontsize=10)
        
        ax4.set_xlabel('Sample Size (n)')
        ax4.set_ylabel('AUC Score (AIF360)')
        ax4.set_title('Performance vs Sample Size', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('FPR (AIF360)', rotation=270, labelpad=15)
        
        plt.suptitle('Algorithm Comparison: Baseline vs GerryFair vs AIF360', 
                    fontweight='bold', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig('outputs/comparison/algorithm_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
    def create_summary_table(self):
        baseline_fprs = self.gerryfair_results['baseline_fprs']
        # Use highest iteration count available  
        iterations = sorted([int(k) for k in self.gerryfair_results['gerryfair_experiments'].keys()])
        best_iteration = max(iterations)
        gerryfair_fprs = self.gerryfair_results['gerryfair_experiments'][str(best_iteration)]['fprs']
        
        summary_data = []
        
        groups = [
            ('Overall', 'overall'),
            ('Male-Rural', 'male_rural'),
            ('Male-Urban', 'male_urban'), 
            ('Female-Rural', 'female_rural'),
            ('Female-Urban', 'female_urban')
        ]
        
        for group_name, group_key in groups:
            if group_key in self.aif360_data:
                aif360_group_data = self.aif360_data[group_key]
                sample_size = aif360_group_data['n']
            else:
                sample_size = 'N/A'
                
            summary_data.append({
                'Group': group_name,
                'Sample Size': sample_size,
                'Baseline FPR': f"{baseline_fprs[group_key]:.3f}",
                'GerryFair FPR': f"{gerryfair_fprs[group_key]:.3f}",
                'AIF360 FPR': f"{self.aif360_data[group_key]['fpr_new']:.3f}" if group_key in self.aif360_data else 'N/A',
                'AIF360 FNR': f"{self.aif360_data[group_key]['fnr_new']:.3f}" if group_key in self.aif360_data else 'N/A',
                'AIF360 AUC': f"{self.aif360_data[group_key]['auc']:.3f}" if group_key in self.aif360_data else 'N/A'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('outputs/comparison/algorithm_comparison_table.csv', index=False)
        
    def run_full_comparison(self):
        self.create_algorithm_comparison_plot()
        self.create_summary_table()

if __name__ == "__main__":
    comparison = AlgorithmComparison()
    comparison.run_full_comparison()
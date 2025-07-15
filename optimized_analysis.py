import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle
import json
import time
import os
from datetime import datetime
import sys

class OptimizedStudentAnalysis:
    def __init__(self, data_path="clean_data/student_data.csv", cache_dir="cache"):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.df = None
        self.results_cache = {}
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        
    def load_cached_results(self, experiment_name):
        cache_file = os.path.join(self.cache_dir, f"{experiment_name}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
        
    def save_results(self, experiment_name, results):
        cache_file = os.path.join(self.cache_dir, f"{experiment_name}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
    
    def load_and_preprocess_data_optimized(self):
        cached_data = self.load_cached_results("preprocessed_data")
        if cached_data is not None:
            self.__dict__.update(cached_data)
            return
            
        self.df = pd.read_csv(self.data_path)
        self.df['female'] = (self.df['sex'] == 'F').astype(int)
        self.df['urban'] = (self.df['address'] == 'U').astype(int)
        self.df['gender_group'] = self.df['sex']
        self.df['location_group'] = self.df['address']
        
        gender_map = {0: 'Male', 1: 'Female'}
        location_map = {0: 'Rural', 1: 'Urban'}
        self.df['intersectional_group'] = (
            self.df['female'].map(gender_map) + '-' + 
            self.df['urban'].map(location_map)
        )
        
        feature_cols = [col for col in self.df.columns 
                       if col not in ['G3', 'sex', 'address', 'gender_group', 
                                    'location_group', 'intersectional_group']]
        
        y = self.df['G3'].values
        X = self.df[feature_cols].copy()
        Xp = self.df[['female', 'urban']]
        
        categorical_cols = X.select_dtypes(include='object').columns
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col])
        
        self.X_train, self.X_test, self.Xp_train, self.Xp_test, \
        self.y_train, self.y_test = train_test_split(
            X, Xp, y, test_size=0.2, random_state=42,
            stratify=self.df['intersectional_group']
        )
        
        cache_data = {
            'df': self.df,
            'X_train': self.X_train, 'X_test': self.X_test,
            'Xp_train': self.Xp_train, 'Xp_test': self.Xp_test,
            'y_train': self.y_train, 'y_test': self.y_test
        }
        self.save_results("preprocessed_data", cache_data)
    
    def compute_fpr_by_groups_optimized(self, y_true, y_pred, groups_data):
        def compute_fpr_vectorized(yt, yp):
            if len(yt) == 0:
                return np.nan
            tn = np.sum((yt == 0) & (yp == 0))
            fp = np.sum((yt == 0) & (yp == 1))
            return fp / (fp + tn) if (fp + tn) > 0 else np.nan
        
        results = {}
        results['overall'] = compute_fpr_vectorized(y_true, y_pred)
        
        female_mask = groups_data['female'] == 1
        male_mask = ~female_mask
        urban_mask = groups_data['urban'] == 1
        rural_mask = ~urban_mask
        
        results['female'] = compute_fpr_vectorized(y_true[female_mask], y_pred[female_mask])
        results['male'] = compute_fpr_vectorized(y_true[male_mask], y_pred[male_mask])
        results['urban'] = compute_fpr_vectorized(y_true[urban_mask], y_pred[urban_mask])
        results['rural'] = compute_fpr_vectorized(y_true[rural_mask], y_pred[rural_mask])
        
        masks = {
            'female_urban': female_mask & urban_mask,
            'female_rural': female_mask & rural_mask,
            'male_urban': male_mask & urban_mask,
            'male_rural': male_mask & rural_mask
        }
        
        for group, mask in masks.items():
            results[group] = compute_fpr_vectorized(y_true[mask], y_pred[mask])
        
        return results
    
    def train_baseline_model_fast(self):
        cached_results = self.load_cached_results("baseline_model")
        if cached_results is not None:
            return cached_results
            
        start_time = time.time()
        baseline = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
        baseline.fit(self.X_train, self.y_train)
        y_pred_baseline = baseline.predict(self.X_test)
        
        baseline_fprs = self.compute_fpr_by_groups_optimized(
            self.y_test, y_pred_baseline, self.Xp_test
        )
        
        results = {
            'model': baseline,
            'predictions': y_pred_baseline,
            'fprs': baseline_fprs,
            'training_time': time.time() - start_time
        }
        
        self.save_results("baseline_model", results)
        return results
    
    def train_gerryfair_optimized(self, max_iters=5):
        cache_key = f"gerryfair_model_iters_{max_iters}"
        cached_results = self.load_cached_results(cache_key)
        if cached_results is not None:
            return cached_results
            
        start_time = time.time()
        sys.path.insert(0, os.getcwd())
        from functions.build_models import gerryfair_model
        
        gerryfair = gerryfair_model(
            self.X_train, self.Xp_train, self.y_train, 
            iters=max_iters, gamma=0.01
        )
        
        y_pred_gerryfair = (gerryfair.predict(self.X_test).values.flatten() >= 0.5).astype(int)
        gerryfair_fprs = self.compute_fpr_by_groups_optimized(
            self.y_test, y_pred_gerryfair, self.Xp_test
        )
        
        results = {
            'model': gerryfair,
            'predictions': y_pred_gerryfair,
            'fprs': gerryfair_fprs,
            'training_time': time.time() - start_time,
            'iterations': max_iters
        }
        
        self.save_results(cache_key, results)
        return results
    
    def run_experiment_batch(self, iteration_counts=[5, 10, 25, 50]):
        self.load_and_preprocess_data_optimized()
        baseline_results = self.train_baseline_model_fast()
        
        all_results = {
            'baseline': baseline_results,
            'gerryfair_experiments': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for iters in iteration_counts:
            gerryfair_results = self.train_gerryfair_optimized(max_iters=iters)
            all_results['gerryfair_experiments'][iters] = gerryfair_results
            self.save_results("batch_experiment_results", all_results)
        
        results_file = os.path.join(self.cache_dir, f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        json_results = {
            'baseline_fprs': baseline_results['fprs'],
            'baseline_training_time': baseline_results['training_time'],
            'gerryfair_experiments': {}
        }
        
        for iters, results in all_results['gerryfair_experiments'].items():
            json_results['gerryfair_experiments'][iters] = {
                'fprs': results['fprs'],
                'training_time': results['training_time'],
                'iterations': results['iterations']
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return all_results
    
    def generate_comparison_plots(self, results_data=None):
        if results_data is None:
            cached_results = self.load_cached_results("batch_experiment_results")
            if cached_results is None:
                return
            results_data = cached_results
        
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        iterations = list(results_data['gerryfair_experiments'].keys())
        training_times = [results_data['gerryfair_experiments'][i]['training_time'] for i in iterations]
        
        ax1.plot(iterations, training_times, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Iterations')
        ax1.set_ylabel('Training Time (seconds)')
        ax1.set_title('GerryFair Training Time vs Iterations')
        ax1.grid(True, alpha=0.3)
        
        baseline_bias = (results_data['baseline']['fprs']['female'] - 
                        results_data['baseline']['fprs']['male'])
        
        gerryfair_biases = []
        for iters in iterations:
            fprs = results_data['gerryfair_experiments'][iters]['fprs']
            bias = fprs['female'] - fprs['male']
            gerryfair_biases.append(bias)
        
        ax2.axhline(y=baseline_bias, color='red', linestyle='--', 
                   label=f'Baseline (bias={baseline_bias:.3f})', linewidth=2)
        ax2.plot(iterations, gerryfair_biases, 'o-', color='blue', 
                label='GerryFair', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Iterations')
        ax2.set_ylabel('Gender FPR Bias')
        ax2.set_title('Gender Bias Reduction vs Iterations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    analysis = OptimizedStudentAnalysis()
    
    small_test_results = analysis.run_experiment_batch([2, 5])
    analysis.generate_comparison_plots(small_test_results)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--long":
        long_results = analysis.run_experiment_batch([10, 25, 50, 100, 200])
        analysis.generate_comparison_plots(long_results)
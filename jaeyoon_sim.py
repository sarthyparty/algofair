import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Base rate and performance simulation
def simulate_fpr(g1_n, g2_n, g1_actual_outcome_p, g2_actual_outcome_p, g1_target_fpr, g2_target_fpr):
    # Generate actual values for groups
    g1_actual_values = np.random.choice([1, 0], size=g1_n, p=[g1_actual_outcome_p, 1 - g1_actual_outcome_p])
    g2_actual_values = np.random.choice([1, 0], size=g2_n, p=[g2_actual_outcome_p, 1 - g2_actual_outcome_p])

    # Function to generate predictions
    def generate_predictions(actual_values, target_fpr):
        negatives = np.sum(actual_values == 0)
        num_false_positives = round(target_fpr * negatives)
        prediction_values = actual_values.copy()
        false_positive_indices = np.random.choice(np.where(actual_values == 0)[0], size=num_false_positives, replace=False)
        prediction_values[false_positive_indices] = 1
        return prediction_values

    # Generate predictions for both groups
    g1_prediction_values = generate_predictions(g1_actual_values, g1_target_fpr)
    g2_prediction_values = generate_predictions(g2_actual_values, g2_target_fpr)

    # Calculate confusion matrices and FPRs
    g1_cm = confusion_matrix(g1_actual_values, g1_prediction_values)
    g2_cm = confusion_matrix(g2_actual_values, g2_prediction_values)
    g1_actual_fpr = g1_cm[1, 0] / (g1_cm[0, 0] + g1_cm[1, 0])
    g2_actual_fpr = g2_cm[1, 0] / (g2_cm[0, 0] + g2_cm[1, 0])

    # Combined data
    total_actual_values = np.concatenate([g1_actual_values, g2_actual_values])
    total_prediction_values = np.concatenate([g1_prediction_values, g2_prediction_values])
    total_cm = confusion_matrix(total_actual_values, total_prediction_values)
    total_fpr = total_cm[1, 0] / (total_cm[0, 0] + total_cm[1, 0])

    # FPR differences
    fpr_diff_g1 = total_fpr - g1_actual_fpr
    fpr_diff_g2 = total_fpr - g2_actual_fpr

    return pd.DataFrame({
        'g1_n': [g1_n],
        'g1_actual_outcome_p': [g1_actual_outcome_p],
        'g1_target_fpr': [g1_target_fpr],
        'g1_actual_fpr': [g1_actual_fpr],
        'g2_n': [g2_n],
        'g2_actual_outcome_p': [g2_actual_outcome_p],
        'g2_target_fpr': [g2_target_fpr],
        'g2_actual_fpr': [g2_actual_fpr],
        'total_fpr': [total_fpr],
        'fpr_diff_g1': [fpr_diff_g1],
        'fpr_diff_g2': [fpr_diff_g2]
    })

# Function to run the simulation
def run_simulation(g1_n_values, g2_n_values, g1_actual_outcome_p_values, g2_actual_outcome_p_values, g1_target_fpr_values, g2_target_fpr_values, file_name):
    # Check if the file exists, if not, create it with header
    if not pd.io.common.file_exists(file_name):
        first_params = simulate_fpr(g1_n_values[0], g2_n_values[0], g1_actual_outcome_p_values[0], g2_actual_outcome_p_values[0], g1_target_fpr_values[0], g2_target_fpr_values[0])
        first_params.to_csv(file_name, index=False, header=True)

    # Iterate over each parameter combination
    for g1_n in g1_n_values:
        for g2_n in g2_n_values:
            for g1_actual_outcome_p in g1_actual_outcome_p_values:
                for g2_actual_outcome_p in g2_actual_outcome_p_values:
                    for g1_target_fpr in g1_target_fpr_values:
                        for g2_target_fpr in g2_target_fpr_values:
                            # Run simulation with these parameters
                            sim_result = simulate_fpr(g1_n, g2_n, g1_actual_outcome_p, g2_actual_outcome_p, g1_target_fpr, g2_target_fpr)

                            # Append each result to the CSV file
                            sim_result.to_csv(file_name, index=False, header=False, mode='a')

# simulate
g1_n_values = np.arange(100, 501, 10)
g2_n_values = np.arange(100, 501, 10)
g1_actual_outcome_p_values = np.arange(0.1, 0.9, 0.1)
g2_actual_outcome_p_values = np.arange(0.1, 0.9, 0.1)
g1_target_fpr_values = np.arange(0.05, 1, 0.1)
g2_target_fpr_values = np.arange(0.05, 1, 0.1)

run_simulation(g1_n_values, g2_n_values, g1_actual_outcome_p_values, g2_actual_outcome_p_values, g1_target_fpr_values, g2_target_fpr_values, "simulation_results_fpr_dec12.csv")

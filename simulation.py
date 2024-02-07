import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

# Function to simulate AUC for a given ratio
def simulate_fpr(ratio):

    n1 = int(0.2 * 1000)
    n2 = int((1 - 0.8) * 1000)
    g1_actual_outcome_p = ratio
    g2_actual_outcome_p = 0.9

    g1_actual_values = np.random.choice([1, 0], size=(n1,), p=[g1_actual_outcome_p, 1 - g1_actual_outcome_p])
    g2_actual_values = np.random.choice([1, 0], size=(n2,), p=[g2_actual_outcome_p, 1 - g2_actual_outcome_p])

    g1_target_fpr = 0.1
    g1_negatives = np.sum(g1_actual_values == 0)
    g1_num_false_positives = round(g1_target_fpr * g1_negatives)

    g1_prediction_values = g1_actual_values.copy()
    g1_false_positive_indices = np.random.choice(np.where(g1_actual_values == 0)[0], size=g1_num_false_positives, replace=False)
    g1_prediction_values[g1_false_positive_indices] = 1

    g2_target_fpr = 0.3
    g2_negatives = np.sum(g2_actual_values == 0)
    g2_num_false_positives = round(g2_target_fpr * g2_negatives)

    g2_prediction_values = g2_actual_values.copy()
    g2_false_positive_indices = np.random.choice(np.where(g2_actual_values == 0)[0], size=g2_num_false_positives, replace=False)
    g2_prediction_values[g2_false_positive_indices] = 1

    # Calculate FPR
    conf_matrix = confusion_matrix(g1_actual_values, g1_prediction_values)
    tn, fp, fn, tp = conf_matrix.ravel()
    fpr = fp / (fp + tn)

    return fpr

# Vary ratio values
ratio_values = np.arange(0.01, 1, 0.01)

# Simulate FPRs for different ratio values
fpr_values = [simulate_fpr(ratio) for ratio in ratio_values]

# Plot FPR change over change in ratio
plt.plot(ratio_values, fpr_values, marker='o')
plt.xlabel('Ratio of 1s')
plt.ylabel('FPR')
plt.title('Change in FPR over Change in Ratio')
plt.grid(True)
plt.show()

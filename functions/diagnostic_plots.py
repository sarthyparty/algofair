from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from functions.build_models import gerryfair_model
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt




def generate_diagnostics(X, y, subgroup_names, demographics, protected, omit_demographics=False, is_gerryfair = False, iters=5, gamma=.01, orig_df=None):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    to_drop = []

    for d in demographics:
        to_drop.append(X.columns.get_loc(d))

    aucs = [[] for _ in subgroup_names]
    fprs = [[] for _ in subgroup_names]
    rmses = [[] for _ in subgroup_names]

    # dataset, attributes = generate_preprocessed.create_attributes(X, y, demographics)
    # X, X_prime, y = gerryfair.clean.clean_dataset(dataset, attributes, False)
    X_prime = X.loc[:, protected]
    X_protected = X.loc[:, demographics]
    if omit_demographics:
        X = X.drop(demographics, axis=1)

    subgroups = []

    for sg in subgroup_names:
        inds = []
        for n in sg:
            inds.append(X_protected.columns.get_loc(n))
        subgroups.append(inds)

    X_cols, X_prime_cols = X.columns, X_prime.columns
    X, X_prime, X_protected, y = np.array(X), np.array(X_prime), np.array(X_protected), np.array(y)

    orig_df_cols = orig_df.columns
    orig_df = np.array(orig_df)
    
    combined_feature = np.dot(X_protected, 2 ** np.arange(X_protected.shape[1])[::-1])

    subgroup_sizes = np.zeros(len(subgroup_names))
    counts = np.zeros(len(subgroup_names))

    fold = 1

    # Iterate through the each fold
    for train_index, test_index in kfold.split(X, combined_feature):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_prime_train, X_prime_test = X_prime[train_index], X_prime[test_index]
        X_protected_train, X_protected_test = X_protected[train_index], X_protected[test_index]

        orig_df_train = orig_df[train_index]

        X_train_df = pd.DataFrame(X_train, columns=X_cols)
        X_prime_train_df = pd.DataFrame(X_prime_train, columns=X_prime_cols)
        y_train_df = pd.Series(y_train)

        model = gerryfair_model(X_train_df, X_prime_train_df, y_train_df, iters, gamma)

        final_df = pd.DataFrame(orig_df_train, columns=orig_df_cols)
        it = pd.read_csv('./diagnostic/black_female_proc/diagnostic.csv')
        final_df = pd.concat([final_df, it], axis=1, join='inner')
        final_df = final_df.drop('Unnamed: 0', axis=1)
        final_df.to_csv(f'diagnostic/black_female_proc/final_fold{fold}.csv', index=False)

        fold += 1


    return []

def gf_learning_curve(X, y, subgroup_names, demographics, protected, omit_demographics=False, is_gerryfair = False, iters=5, gamma=.01):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    to_drop = []

    for d in demographics:
        to_drop.append(X.columns.get_loc(d))

    aucs = [[] for _ in subgroup_names]
    fprs = [[] for _ in subgroup_names]
    rmses = [[] for _ in subgroup_names]

    # dataset, attributes = generate_preprocessed.create_attributes(X, y, demographics)
    # X, X_prime, y = gerryfair.clean.clean_dataset(dataset, attributes, False)
    X_prime = X.loc[:, protected]
    X_protected = X.loc[:, demographics]
    if omit_demographics:
        X = X.drop(demographics, axis=1)

    subgroups = []

    for sg in subgroup_names:
        inds = []
        for n in sg:
            inds.append(X_protected.columns.get_loc(n))
        subgroups.append(inds)

    X_cols, X_prime_cols = X.columns, X_prime.columns
    X, X_prime, X_protected, y = np.array(X), np.array(X_prime), np.array(X_protected), np.array(y)
    
    combined_feature = np.dot(X_protected, 2 ** np.arange(X_protected.shape[1])[::-1])

    fold = 1

    training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def select_subset(data, size):
        # indices = np.random.choice(len(data), size=int(size * len(data)), replace=False)
        # subset = [data[i] for i in indices]
        # return subset
        return data[:int(size * len(data))]



    for train_index, test_index in kfold.split(X, combined_feature):

        auc_scores = []

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_prime_train, X_prime_test = X_prime[train_index], X_prime[test_index]
        X_protected_train, X_protected_test = X_protected[train_index], X_protected[test_index]

        model = None
        X_train_df = pd.DataFrame(X_train, columns=X_cols)
        X_test_df = pd.DataFrame(X_test, columns=X_cols)
        X_prime_train_df = pd.DataFrame(X_prime_train, columns=X_prime_cols)
        y_train_df = pd.Series(y_train)
        y_test_df = pd.Series(y_test)
        
        for size in training_sizes:
            # Select subset of training data
            X_train_subset = select_subset(X_train, size)
            X_prime_train_subset = select_subset(X_prime_train, size)
            y_train_subset = select_subset(y_train, size)

            X_train_subset_df = pd.DataFrame(X_train_subset, columns=X_cols)
            X_prime_train_subset_df = pd.DataFrame(X_prime_train_subset, columns=X_prime_cols)
            y_train_subset_df = pd.Series(y_train_subset)

            model = gerryfair_model(X_train_subset_df, X_prime_train_subset_df, y_train_subset_df, iters, gamma)
            
            # Evaluate model on testing dataset
            predictions = model.predict(X_test_df)
            auc = roc_auc_score(y_test, predictions)
            
            # Store AUC score
            auc_scores.append(auc)

        # Plot learning curve
        x_axis = [int(size * len(X_train)) for size in training_sizes]
        # save auc_scores to file
        np.array(auc_scores).tofile(f'learning_curves/auc_scores_fold_{fold}.csv', sep=',')

        plt.plot(x_axis, auc_scores)
        plt.xlabel('Training Set Size')
        plt.ylabel('AUC Score')
        plt.title(f'Learning Curve Fold {fold}')
        plt.show()
        
        
        fold += 1
    

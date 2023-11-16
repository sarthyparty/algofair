from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
import pandas as pd
import gerryfair
from functions.build_models import gerryfair_model, logr_model
from sklearn.model_selection import StratifiedKFold
from functions.formatting import get_indices, get_subgroup_str

def calc_metrics(X, y, subgroups_dict, demographics, omit_demographics=False, is_gerryfair = False):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    subgroups, names = get_indices(subgroups_dict, X)
    # subgroups = subgroups[:5]
    # names = names[:5]

    to_drop = []

    for d in demographics:
        to_drop.append(X.columns.get_loc(d))

    aucs = [[] for _ in subgroups]
    fprs = [[] for _ in subgroups]
    rmses = [[] for _ in subgroups]

    # dataset, attributes = generate_preprocessed.create_attributes(X, y, demographics)
    # X, X_prime, y = gerryfair.clean.clean_dataset(dataset, attributes, False)
    X_prime = X.loc[:, demographics]

    X_cols, X_prime_cols = X.columns, X_prime.columns
    X, X_prime, y = np.array(X), np.array(X_prime), np.array(y)
    
    combined_feature = np.dot(X_prime, 2 ** np.arange(X_prime.shape[1])[::-1])

    subgroup_sizes = np.zeros(len(subgroups))
    counts = np.zeros(len(subgroups))
    # Iterate through the each fold
    for train_index, test_index in kfold.split(X, combined_feature):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_prime_train, X_prime_test = X_prime[train_index], X_prime[test_index]

        model = None

        if is_gerryfair:
            X_train_df = pd.DataFrame(X_train, columns=X_cols)
            X_prime_train_df = pd.DataFrame(X_prime_train, columns=X_prime_cols)
            y_train_df = pd.Series(y_train)

            model = gerryfair_model(X_train_df, X_prime_train_df, y_train_df)
        else:
            model = logr_model(X_train, y_train)
        
        for i, subgroup in enumerate(subgroups):
            X_test_filtered = X_test
            y_test_filtered = y_test
            if subgroup:
                conditions = np.array([(X_test_filtered[:, name] == val) for name, val in subgroup]).all(axis=0)
                X_test_filtered = X_test_filtered[conditions]
                y_test_filtered = y_test_filtered[conditions]

            subgroup_sizes[i] += X_test_filtered.shape[0]
            counts[i]+=1

            # if omit_demographics and not is_gerryfair:
            #     X_train = np.delete(X_train, to_drop, axis=1)    
            #     X_test_filtered = np.delete(X_test_filtered, to_drop, axis=1)        

            res = None
            if is_gerryfair:
                X_test_df = pd.DataFrame(X_test_filtered, columns=X_cols)
                res = calc_metric(model, X_test_df, y_test_filtered, True)
            else:
                res = calc_metric(model, X_test_filtered, y_test_filtered, False)
        
        
            if res == None:
                continue

            auc, FPR, rmse = res
            aucs[i].append(auc)
            fprs[i].append(FPR)
            rmses[i].append(rmse)

    ret = []

    for i, subgroup in enumerate(names):
        subgroup_data = {
            'subgroup': get_subgroup_str(subgroup),
            'n': f"{subgroup_sizes[i]/counts[i]}",
            'auc_avg': f"{np.average(aucs[i]):.3f}",
            'auc_std': f"{np.std(aucs[i]):.3f}",
            'fpr_avg': f"{np.average(fprs[i]):.3f}",
            'fpr_std': f"{np.std(fprs[i]):.3f}",
            'rmse_avg': f"{np.average(rmses[i]):.3f}",
            'rmse_std': f"{np.std(rmses[i]):.3f}"
        }

        ret.append(subgroup_data)
    
    # print(f"AUC: {auc_avg} +/- {auc_std}")
    # print(f"FPR: {fpr_avg} +/- {fpr_std}")
    return ret

def calc_metric(model, X_test, y_test, is_gerryfair):
    auc = None
    rmse = None
    y_pred = None
    try:
        y_pred = np.array(model.predict(X_test))
        if is_gerryfair:
            y_pred = y_pred.ravel()
        # if is_gerryfair:
        #     print(y_pred)
        #     y_pred = (y_pred.values >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    except Exception:
        print('error')
        return None
    y_test = np.array(y_test)
    TN = np.sum((y_test == 0) & (y_pred == 0))
    FP = np.sum((y_test == 0) & (y_pred == 1))      
    FPR = FP / (FP + TN)

    print(y_test, y_pred, FPR)

    return auc, FPR, rmse



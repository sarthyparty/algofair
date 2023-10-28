from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
from functions.build_models import gerryfair_model, logr_model
from sklearn.model_selection import KFold

def calc_metrics(X, y, subgroups, demographics=None, omit_demographics=False, gerryfair = False):
    col_subgroups = subgroups.copy()

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # Gets index of given columns
    for i,d in enumerate(col_subgroups):
        col_subgroups[i] = (X.columns.get_loc(d[0]), d[1])

    to_drop = []
    if demographics == None:
        demographics = ['country_cd_US', 'is_female', 'bachelor_obtained', 'white']

    for d in demographics:
        to_drop.append(X.columns.get_loc(d))

    aucs = []
    fprs = []
    rmses = []

    y = np.array(y)
    X = np.array(X)

    subgroup_size = 0
    count = 0
    # Iterate through the each fold
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if subgroups:
            conditions = np.array([(X_test[:, name] == val) for name, val in col_subgroups]).all(axis=0)
            X_test = X_test[conditions]
            y_test = y_test[conditions]

        X_prime = X_train[[i[0] for i in col_subgroups]]

        subgroup_size += X_test.shape[0]
        count+=1

        if omit_demographics:
            X_train = np.delete(X_train, to_drop, axis=1)                
        
        if omit_demographics:
            X_test = np.delete(X_test, to_drop, axis=1)

        model = None
        if gerryfair:
            model = gerryfair_model(X_train, X_prime, y_train)
        else:
            model = logr_model(X_train, y_train)
        
        res = calc_metric(model, X_test, y_test)
        if res == None:
            continue

        auc, FPR, rmse = res
        aucs.append(auc)
        fprs.append(FPR)
        rmses.append(rmse)

    auc_avg = np.average(aucs)
    auc_std = np.std(aucs)
    fpr_avg = np.average(fprs)
    fpr_std = np.std(fprs)
    rmse_avg = np.average(rmses)
    rmse_std = np.std(rmses)
    
    # print(f"AUC: {auc_avg} +/- {auc_std}")
    # print(f"FPR: {fpr_avg} +/- {fpr_std}")
    return (auc_avg, auc_std, fpr_avg, fpr_std, rmse_avg, rmse_std, subgroup_size/count)

def calc_metric(model, X_test, y_test):
    auc = None
    rmse = None
    y_pred = None
    try:
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    except Exception:
        return None
    
    TN = np.sum((y_test == 0) & (y_pred == 0))
    FP = np.sum((y_test == 0) & (y_pred == 1))      
    FPR = FP / (FP + TN)

    return auc, FPR, rmse

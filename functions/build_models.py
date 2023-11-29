import warnings
from sklearn.linear_model import LogisticRegression
import gerryfair

def gerryfair_model(X_train, X_prime_train, y_train, iters, gamma):
    warnings.filterwarnings("ignore", category=UserWarning)

    C = 15
    printflag = False
    fair_model = gerryfair.model.Model(C=C, printflag=printflag, gamma=gamma)
    max_iters = iters
    fair_model.set_options(max_iters=max_iters)

    # Train the model
    [errors, fp_difference] = fair_model.train(X_train, X_prime_train, y_train)
    return fair_model

def logr_model(X_train, y_train):
    model = LogisticRegression(verbose=0)
    model.fit(X_train, y_train)
    return model

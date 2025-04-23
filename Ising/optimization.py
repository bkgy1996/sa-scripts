import numpy as np
from itertools import product
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def data_prepare(ps:np.ndarray,tags:np.ndarray,weights:np.ndarray,percentage=50):
    subject_num = ps.shape[0]
    eta_num = ps.shape[1]
    if len(tags.shape)!= 1 or tags.shape[0]!=subject_num:
        raise ValueError('tags mismatch')
    if len(weights.shape)!=1 or weights.shape[0]!=eta_num:
        raise ValueError('weights mismatch')
    _weights = weights.copy()
    _weights[_weights<np.percentile(_weights,percentage)] = 0
    part_weights = _weights[_weights!=0]
    part_ps = ps[:,_weights!=0]
    part_weights = part_weights[np.newaxis,:]
    w_ps = part_ps*part_weights
    return w_ps,tags,_weights


def sign_weight_cross_val(X, y, n_splits=5, random_state=12, fit_intercept=True):
    """
    1. In each CV fold, fit a normal linear regression on the training set.
    2. Round the learned coefficients to ±1.
    3. Use (optionally) the real intercept or no intercept at all.
    4. Evaluate R^2 on the test set with these ±1 weights.

    Returns the mean and std of R^2 across folds.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    rs = []
    ps = []
    ws = []
    for train_idx, test_idx in kf.split(X):
        # X_train, X_test = X[train_idx], X[test_idx]
        # y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test = X[test_idx], X[train_idx]
        y_train, y_test = y[test_idx], y[train_idx]
        # 1) Fit an unconstrained linear model on training fold
        reg = LinearRegression(fit_intercept=fit_intercept)
        reg.fit(X_train, y_train)

        # 2) Round coefficients to ±1
        w_real = reg.coef_  # real-valued weights
        w_sign = np.where(w_real >= 0, 1.0, -1.0)  # ±1 weights
        ws.append(w_sign)
        # 3) (Optional) Keep the learned intercept
        if fit_intercept:
            intercept = reg.intercept_
        else:
            intercept = 0.0

        # 4) Predict on the test fold using the ±1 weights
        y_pred = intercept + X_test.dot(w_sign)

        # Compute R^2
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        scores.append(r2)
        # Compute pearson
        r,p = pearsonr(y_test,y_pred)
        rs.append(r)
        ps.append(p)

    return np.mean(scores), ws,rs,ps

def evaluate_sign_vector(sign_vector, X, y, n_splits=5):
    """
    Multiplies each column of X by the corresponding sign from sign_vector,
    then performs n-fold cross validation on the transformed data.

    Returns the mean cross-validation accuracy.
    """
    # Transform X by applying ±1 sign to each column
    X_transformed = X * sign_vector

    # Use logistic regression for demonstration
    model = LinearRegression()

    # 5-fold cross validation
    kf = KFold(n_splits=n_splits, shuffle=True)

    # scoring='r2' by default for regression in cross_val_score
    # but let's be explicit:
    scores = cross_val_score(model, X_transformed, y, cv=kf, scoring='r2')


    return np.mean(scores)


# -----------------------
# 3. Brute-force all possible ±1 assignments
# -----------------------
def brute_force_find(X,y,k_fold):
    n_features = X.shape[1]
    best_sign_vector = None
    best_score = -np.inf

    # Each sign_vector is an element of {+1, -1}^N
    for signs in product([-1, 1], repeat=n_features):
        sign_vector = np.array(signs)
        score = evaluate_sign_vector(sign_vector, X, y, n_splits=k_fold)

        if score > best_score:
            best_score = score
            best_sign_vector = sign_vector
    return best_score,best_sign_vector

if __name__ == '__main__':
    pass
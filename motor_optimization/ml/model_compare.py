
# ml/model_compare.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score

def compare_models(seed=42):
    X = np.random.rand(600, 6)
    y = 0.6 + 0.25*np.mean(X, axis=1) + 0.1*np.random.rand(600)
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=seed),
        "Gaussian Process": GaussianProcessRegressor()
    }
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        results[name] = (-scores.mean(), scores.std())
    return results

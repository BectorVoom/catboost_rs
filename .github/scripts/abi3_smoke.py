"""Fit/predict smoke test used by python-wheels.yml to prove the abi3-py312
wheel actually works on newer CPython minors (3.13, 3.14), not just the 3.12
interpreter it was built against."""
import numpy as np
import catboost_rs

X = np.random.RandomState(0).rand(50, 4).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 1.0).astype(np.float32)
m = catboost_rs.CatBoostRegressor(iterations=5)
m.fit(X, y)
m.predict(X[:5])
print("fit/predict OK on", catboost_rs.__name__)

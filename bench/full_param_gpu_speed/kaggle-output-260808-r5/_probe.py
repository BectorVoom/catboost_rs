import os, sys, json, numpy as np, catboost_rs
kw = json.loads("{\"iterations\": 30, \"depth\": 6, \"learning_rate\": 0.1, \"l2_leaf_reg\": 3.0, \"border_count\": 32, \"random_seed\": 42, \"random_strength\": 0, \"bootstrap_type\": \"Bernoulli\", \"boost_from_average\": false, \"leaf_estimation_method\": \"Gradient\", \"score_function\": \"L2\", \"loss_function\": \"RMSE\", \"grow_policy\": \"Depthwise\", \"subsample\": 0.66}")
kw['iterations'] = 2
n_rows, n_features = (300000, 50)
sys.path.insert(0, "/tmp/repo/bench")
import generator as gen
X, yr = gen.generate(n_rows, n_features, seed=42)
ctr = False
cat = gen.generate_cat(n_rows, seed=42) if ctr else None
kind = "reg"
y = yr if kind == 'reg' else (gen.cat_driven_binary_target(X, cat, seed=42) if ctr else gen.binary_target(X, seed=42))
w = gen.generate_weights(n_rows) if False else None
Cls = catboost_rs.CatBoostRegressor if kind == 'reg' else catboost_rs.CatBoostClassifier
m = Cls(**kw)
fit_kw = {}
if w is not None:
    fit_kw['sample_weight'] = w
if ctr:
    import numpy as np
    Xf = np.concatenate([X, cat.astype(X.dtype)], axis=1)
    fit_kw['cat_features'] = list(range(X.shape[1], Xf.shape[1]))
    m.fit(Xf, y, **fit_kw)
else:
    m.fit(X, y, **fit_kw)

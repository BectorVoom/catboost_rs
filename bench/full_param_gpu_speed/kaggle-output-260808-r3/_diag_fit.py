import os, sys, json, time, numpy as np, catboost_rs
kw = json.loads("{\"iterations\": 30, \"depth\": 6, \"learning_rate\": 0.1, \"l2_leaf_reg\": 3.0, \"border_count\": 32, \"random_seed\": 42, \"random_strength\": 0, \"bootstrap_type\": \"No\", \"boost_from_average\": false, \"leaf_estimation_method\": \"Gradient\", \"score_function\": \"L2\", \"loss_function\": \"RMSE\", \"grow_policy\": \"SymmetricTree\"}")
n_rows, n_features = (1000000, 50)
sys.path.insert(0, "/tmp/repo/bench")
import generator as gen
X, y = gen.generate(n_rows, n_features, seed=42)
m = catboost_rs.CatBoostRegressor(**kw)
t0 = time.time()
m.fit(X, y)
print(f'WALLCLOCK_FIT_SECONDS={time.time() - t0:.4f}', flush=True)
_ = m.predict(X[:1024])

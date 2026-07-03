
## 11-05 deferred (out-of-scope, pre-existing)

- **cell 6 (depth-1 Logloss oracle) uses `clf.predict(X, prediction_type='RawFormulaVal')`**
  but `catboost-rs-py` `CatBoostClassifier.predict` takes only `(x)` (no `prediction_type`
  kwarg) and returns class labels; the raw margin comes from `predict_proba(X)[:,1]` → logit.
  This is a pre-existing depth-1 harness bug (Plan 10 era), NOT caused by 11-05. Left
  byte-unchanged per D-04 (do not alter the depth-1 CPU reference flow). The 11-05 depth-6
  cells are self-contained (re-import the wheel, use `predict_proba`→logit) and can be run
  independently of the depth-1 cells. Fix cell 6 in a depth-1 harness follow-up.

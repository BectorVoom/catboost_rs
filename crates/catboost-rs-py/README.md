# catboost-rs (Python bindings)

PyO3 bindings exposing the Rust-native `catboost-rs` gradient-boosting library as
the `catboost_rs` Python module — both scikit-learn-compatible and CatBoost-native.

> Phase 8 / plan 08-01 stands up the thinnest end-to-end vertical slice:
> `CatBoostRegressor().fit(X, y).predict(X)` over a float32 contiguous NumPy
> array, wired through the real facade. Param validation, multi-source ingestion,
> the classifier / ranker, the error taxonomy, and packaging land in later plans.

## Backends

- Default wheel: **cpu** (abi3-py312, covers CPython 3.12/3.13/3.14 GIL builds).
- ROCm GPU: separate distribution `catboost-rs-rocm`, pulled via the `[rocm]`
  extra. Both wheels expose the `catboost_rs` import name and must not be
  installed simultaneously.

## Development

```bash
python3 -m venv .venv-py8
.venv-py8/bin/pip install "maturin>=1.9.4,<2.0" scikit-learn numpy pandas pyarrow polars
.venv-py8/bin/maturin develop --features cpu
.venv-py8/bin/pytest tests -q
```

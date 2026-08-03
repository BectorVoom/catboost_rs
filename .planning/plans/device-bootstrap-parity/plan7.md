---
plan: 7
task_id: TASK-07
phase: device-bootstrap-parity
status: pending
order: 7
wave: A
hardware: none required (Python + CPU oracle)
depends_on: [none]
specifications: [WR01-S14]
---

# Task 7: Generate the bias-0 upstream fixture family and gate it on the CPU path

## Objective

After this task `crates/cb-oracle/fixtures/bootstrap_dev/{no,bayesian,bernoulli,mvs}/`
exists — the committed `bootstrap/` family regenerated with the single change
`boost_from_average=False` — and a CPU oracle proves the Rust CPU grower matches it
at ≤1e-5. That proves the FIXTURE before any device claim rests on it, and it makes
the fixtures device-eligible (CR-01 requires `bias == 0.0`,
`boosting.rs:3092`).

This task has no GPU dependency and no dependency on any other task, so it belongs
in Wave A: it can be running while TASK-01/05 proceed.

## Specification references

- `WR01-S14` — a bias-0 upstream fixture family exists and gates the CPU path.
  Primary failure reason: the new fixtures encode a different model than the Rust
  side builds ("pin every raw-dict default").

## Context and evidence

- **catboost is available here.** `python3 -c "import catboost"` →
  `1.2.10` on system Python 3.14.6, and the repo's `.venv` also has `1.2.10`
  `[VERIFIED: RUN]`. This retires research Open Question 1 / LOW-confidence item 24.
- **The existing generator.** `gen_bootstrap()` at
  `crates/cb-oracle/generator/gen_fixtures.py:705-830` `[VERIFIED]`:
  - synthesizes and commits `inputs/bootstrap_multiblock` (1500×4, seed
    `INPUT_SEED_BOOTSTRAP = 20260613`) — **REUSE it, do not regenerate**, or every
    committed bootstrap fixture becomes stale;
  - scenarios `[("no","No",{}), ("bayesian","Bayesian",{"bagging_temperature":1.0}),
    ("bernoulli","Bernoulli",{"subsample":0.8}), ("mvs","MVS",{"subsample":0.8})]`;
  - `shared = {**ISOLATING_PARAMS minus bootstrap_type, "iterations": 3,
    "boost_from_average": True}`;
  - writes `model.json`, `staged.npy`, `predictions.npy`, `config.json`.
- **`ISOLATING_PARAMS`** (`gen_fixtures.py:146-159`) already pins the dangerous
  defaults: `random_strength: 0`, `leaf_estimation_iterations: 1`,
  `score_function: "L2"`, `leaf_estimation_method: "Gradient"`, `random_seed: 0`
  (`SEED`), `thread_count: 1`, `learning_rate: 0.1`, `depth: 2`, `l2_leaf_reg: 3.0`
  `[VERIFIED]`. **`random_strength` is the specific trap the user flagged** — it is
  pinned here and must be mirrored on the Rust side.
- **Targeted-entrypoint convention** so committed fixtures are not clobbered:
  `if "--wave1-only" in sys.argv: … elif "--multiquantile-only" … else: main()`
  at `gen_fixtures.py:3246-3272`, with each `gen_*_only()` documenting exactly what
  it rewrites `[VERIFIED]`.
- **The Rust-side oracle to mirror.** `crates/cb-train/tests/bootstrap_oracle_test.rs`
  — `fixture()` path helper, `load_feature_columns()` (X as f32 SoA),
  `load_model_json(...).float_feature_borders()`, the `BoostParams` block
  (`iterations: 3, depth: 2, learning_rate: 0.1, l2_leaf_reg: 3.0,
  random_strength: 0.0, boost_from_average: true, leaf_method: Gradient,
  random_seed: 0, score_function: L2`), and `compare_stage(..., Stage::…)`
  `[VERIFIED: file read this session]`.
- **Poisson is deliberately absent** from the family (upstream rejects it on CPU,
  documented at `gen_fixtures.py:718-722`). Keep it absent.

## Files

- Modify: `crates/cb-oracle/generator/gen_fixtures.py`
  - add `BOOTSTRAP_DEV = FIXTURES / "bootstrap_dev"` next to `BOOTSTRAP`
    (`gen_fixtures.py:61-62`);
  - add `def gen_bootstrap_dev() -> None:` — a near-copy of `gen_bootstrap()` that
    (a) **does not** write `inputs/bootstrap_multiblock` (it LOADS the committed
    `X.npy` / `y.npy`), (b) sets `"boost_from_average": False`, (c) writes under
    `BOOTSTRAP_DEV`, (d) records in `config.json` that this family exists to be
    device-eligible (bias 0) and names the CR-01 gate at `boosting.rs:3092`;
  - add `def gen_bootstrap_dev_only() -> None:` and the
    `elif "--bootstrap-dev-only" in sys.argv:` arm;
  - **do not** call `gen_bootstrap_dev()` from `main()` in this task — keep `main()`
    byte-unchanged so a full regeneration cannot be triggered accidentally. Add a
    TODO-style comment stating that it should be added to `main()` once the family
    is committed and stable.
- Create: `crates/cb-oracle/fixtures/bootstrap_dev/{no,bayesian,bernoulli,mvs}/`
  (`model.json`, `staged.npy`, `predictions.npy`, `config.json`) — generated output.
- Create: `crates/cb-train/tests/bootstrap_dev_oracle_test.rs` — the CPU arm
  (the device arm is added by TASK-08 in the same file).

## TDD sequence

### 1. Red

- Write `bootstrap_dev_oracle_test.rs` FIRST, before generating anything. It is a
  near-copy of `bootstrap_oracle_test.rs` with two changes: the fixture prefix is
  `bootstrap_dev/…` and `boost_from_average: false`. It must fail with a missing-file
  error — that is the RED state, and it forces the fixture path contract to be
  written down before the generator runs.
- Add one extra test not present in the original:
  `bias_is_zero_in_every_bootstrap_dev_scenario` — load each `model.json` and assert
  its scale/bias record is `0.0`, so a mis-generated fixture (with
  `boost_from_average` accidentally left `True`) fails loudly here rather than
  silently disabling the device path in TASK-08.
- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test`

### 2. Green

1. Add `gen_bootstrap_dev()` + the `--bootstrap-dev-only` arm.
2. Generate:
   `python3 crates/cb-oracle/generator/gen_fixtures.py --bootstrap-dev-only`
   (or `.venv/bin/python`).
3. **Immediately** run `git status --short crates/cb-oracle/fixtures/` and confirm
   the ONLY new/changed paths are under `bootstrap_dev/`. If anything under
   `bootstrap/` or `inputs/` changed, `git checkout` those paths and fix the
   generator before continuing — this is R13 and it is unrecoverable if it lands.
4. Run the CPU oracle; iterate on the Rust `BoostParams` until every stage passes at
   ≤1e-5. If a stage fails, the first suspect is a parameter pinned on one side only
   — diff the generated `config.json["params"]` against the Rust `BoostParams` field
   by field before touching any algorithm.
- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test -- --nocapture`

### 3. Refactor

- Factor the shared body of `gen_bootstrap()` and `gen_bootstrap_dev()` into one
  helper taking `(out_dir, boost_from_average, write_inputs)` so the two families
  can never drift on anything except `boost_from_average`. Regenerate and confirm
  `git status` shows `bootstrap_dev/` unchanged (byte-identical regeneration proves
  determinism) and `bootstrap/` untouched.
- On the Rust side, factor the shared oracle harness (`fixture`,
  `load_feature_columns`, `train_scenario`, `check_scenario`) so
  `bootstrap_oracle_test.rs` and `bootstrap_dev_oracle_test.rs` differ only in the
  prefix and the `boost_from_average` flag. Because Rust integration tests do not
  share a crate, put the harness in `crates/cb-train/tests/common/bootstrap_harness.rs`
  and `mod common;` from both.
- Run: both oracle tests.

### 4. Verify

- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test -- --nocapture`
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` (**blocking** — must be
  green AND unchanged)
- Run: `git status --short crates/cb-oracle/fixtures/bootstrap/` → **empty**
- Run: `git status --short crates/cb-oracle/fixtures/inputs/` → **empty**
- Run: `python3 crates/cb-oracle/generator/gen_fixtures.py --bootstrap-dev-only`
  a second time, then `git status --short crates/cb-oracle/fixtures/bootstrap_dev/`
  → **empty** (deterministic regeneration)
- Run: `cargo test -p cb-train` (CPU suite; one known pre-existing red)
- Confirm: every `bootstrap_dev/*/config.json` records
  `"boost_from_average": false`, `"random_strength": 0`, `"thread_count": 1`,
  `"catboost_version": "1.2.10"`.

## Implementation steps

1. Write the failing Rust oracle (including the bias-is-zero test).
2. Add the generator function + CLI arm.
3. Generate; check `git status` immediately.
4. Iterate to green on the four scenarios (`no` is the control: it must match the
   existing `No`-bootstrap behaviour at bias 0 and is the cheapest one to debug).
5. Refactor both sides to share.
6. Prove regeneration determinism.

## Completion criteria

- [ ] `bootstrap_dev/{no,bayesian,bernoulli,mvs}/` committed with all four artifacts
      each.
- [ ] The CPU oracle passes at ≤1e-5 on splits, leaf values, staged approximants and
      predictions for all four scenarios.
- [ ] Every scenario's model bias is `0.0`.
- [ ] `crates/cb-oracle/fixtures/bootstrap/` and `inputs/` are byte-unchanged.
- [ ] Regeneration is deterministic (second run produces no diff).
- [ ] `bootstrap_oracle_test` green and unchanged.
- [ ] `main()` in the generator is unchanged.

## Risks and guardrails

- **R13 clobbering committed fixtures** — the single biggest risk in this task. Two
  guards: the `--bootstrap-dev-only` entrypoint (never `main()`), and the immediate
  `git status` check in step 3.
- **The `random_strength` trap** — pinned by `ISOLATING_PARAMS` on the Python side
  and by `random_strength: 0.0` on the Rust side. Guard: a field-by-field diff of
  `config.json["params"]` against `BoostParams` is a required debugging step, and
  the harness refactor makes the two families structurally identical.
- **Regenerating the input dataset** would invalidate the committed `bootstrap/`
  family (which reads the same `inputs/bootstrap_multiblock`). Guard:
  `gen_bootstrap_dev()` must LOAD, never `np.save`, the input files; the
  `git status crates/cb-oracle/fixtures/inputs/` check proves it.
- **Python version drift** — the environment has Python 3.14.6 with catboost 1.2.10;
  record both in the generated `config.json` so a future regeneration mismatch is
  diagnosable.
- **Pruned borders** — the oracle reads borders from `model.json`
  (`float_feature_borders()`), which are the model's PRUNED set. That is the
  existing, working convention of `bootstrap_oracle_test.rs`; do not switch to the
  standalone binarizer here.

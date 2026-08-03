---
plan: 9
task_id: TASK-09
phase: device-bootstrap-parity
status: pending
order: 9
wave: E
hardware: local ROCm gfx1151 (sampler oracle) + CPU (error contract)
depends_on: [TASK-06]
specifications: [WR01-S16]
---

# Task 9: Poisson — one backend-independent, self-explaining contract

## Objective

After this task a Poisson fit behaves **identically on every backend** — rejected up
front with a typed error that explains why — and the device Poisson sampler retains
a strengthened *capability* oracle (determinism, in-repo reference agreement,
finiteness) that makes no parity claim. The intentional divergence from the CPU path
is documented in code, not just in a plan.

See `SPEC.md` §8 for the full decision (D5) and its four-point rationale. This task
implements D5. §"Alternative A" below scopes the opposite choice if the user
overrides.

## Specification references

- `WR01-S16` — Poisson has one defined, backend-independent behaviour.
  Primary failure reason: a Poisson fit succeeds on one build and errors on another,
  or produces a device model with no upstream meaning.

## Context and evidence

- **Upstream supports Poisson on GPU only.** The in-repo CPU sampler mirrors that
  by returning
  `Err(CbError::Degenerate("poisson bootstrap is not supported on CPU"))`
  `[VERIFIED: crates/cb-train/src/bootstrap.rs:435-438]`, and the module doc records
  the upstream source `bootstrap_options.cpp:27-33`
  `[VERIFIED: bootstrap.rs:40-43]`.
- **No CatBoost-CPU oracle can exist** — the generator says so explicitly and omits
  the scenario `[VERIFIED: gen_fixtures.py:718-722]`.
- **This project's device path targets CatBoost-CPU semantics**, not CatBoost-GPU's
  algorithm; that is the basis of every device oracle in the repo (see the D-04 /
  T-10-05 restatement at `boosting.rs:3044-3053`).
- **Today's failure point is late and misleading.** A Poisson request walks into the
  boosting loop and fails inside `bootstrap()` on tree 0 with a message that says
  "not supported on CPU" — wrong wording on a GPU build.
- **The device sampler exists and is deterministic-but-arbitrary in its stream
  advance.** `session.rs:1560-1579` documents that Poisson (Knuth) consumes a
  VARIABLE number of draws per object while the session advances by exactly `n`, so
  the phase is "deterministic-but-arbitrary, NOT aligned to the draws actually
  consumed" `[VERIFIED]`. That comment is accurate and must be preserved and
  cross-referenced, not deleted.
- **The typed variant — DECIDED 2026-07-30: KEEP `CbError::Degenerate`.**
  `CbError::Unsupported` would be a better semantic fit ("unsupported on the active
  backend", `[VERIFIED: crates/cb-core/src/error.rs:86-92]`) than `Degenerate`
  ("degenerate training input", `:79-84`), but switching the variant is a BREAKING
  change for any caller matching on `Degenerate` for this case. The user chose the
  lower-blast-radius option: keep `CbError::Degenerate` and fix ONLY the misleading
  message. Every "Unsupported" reference below is superseded by this decision.
- **The gate is an allow-list** after TASK-06, so Poisson already never reaches the
  device path; this task adds the early rejection and the defence-in-depth session
  decline.

## Files

- Modify: `crates/cb-train/src/boosting.rs` — reject Poisson at parameter validation,
  before the device session `begin` and before the iteration loop. Place it with the
  other up-front validations (alongside `check_depth` / `validate_grow_policy`); if
  there is no single validation site, add the check immediately after `params` is
  first inspected in `train_inner` and note why the placement matters.
- Modify: `crates/cb-train/src/bootstrap.rs:435-438` — KEEP the
  `CbError::Degenerate` variant; rewrite ONLY the message to state (a) upstream supports
  Poisson only on GPU, (b) this project's device path reproduces CatBoost **CPU**
  semantics, (c) therefore no ≤1e-5 parity claim is possible and the fit is rejected
  on every backend. Keep the upstream file:line citation in the doc comment.
- Modify: `crates/cb-train/src/bootstrap_test.rs` — update the existing Poisson
  dispatch-error unit test to the new variant/message.
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs` — in the oblivious coverage
  gate (`:1038-1092`) decline explicitly when the mapped bootstrap arm is Poisson,
  with a comment naming WR01-S16 (defence in depth: relaxing the host gate later
  cannot silently enable it). Preserve and cross-reference the accurate
  variable-draw note at `:1567-1577`.
- Modify: `crates/cb-backend/src/kernels/bootstrap_device_test.rs` — strengthen the
  Poisson self-oracle (see Red below) and add a module-level paragraph stating this
  is a CAPABILITY oracle, unreachable from `train()`, with no parity claim.
- Create (if the reference does not already exist in the test file):
  a small host Knuth-Poisson reference inside the test file only.

## TDD sequence

### 1. Red

**A. The error contract** — `crates/cb-train/tests/poisson_contract_test.rs` (new):

- `poisson_fit_is_rejected_with_typed_error` — `train(...)` with
  `bootstrap_type = Poisson` returns `Err(CbError::Degenerate(msg))` where `msg`
  contains "Poisson", "GPU", and "CPU semantics". Principal failure reason: the
  rejection is missing, late, or the wrong variant. NOTE: the message must NOT say
  "not supported on CPU" — that wording is what makes today's failure confusing on a
  device fit (empirically confirmed: upstream CatBoost CPU emits the nearly identical
  string, so the two are indistinguishable in a log).
- `poisson_rejection_precedes_any_training_work` — assert the error is returned for
  a `params` that would ALSO fail later for a different reason (e.g. `iterations = 0`
  plus Poisson), proving the Poisson check runs first and is not reached via the
  sampler. Alternatively assert via a `Runtime` whose `begin_device_training` panics
  if called: a correct implementation never calls it.
- The SAME two tests must be compiled and run under both the default `cpu` build and
  `--no-default-features --features rocm` — same variant, same message. This is the
  backend-independence claim, so the file must have NO `#[cfg(feature)]` gating.

**B. The sampler capability oracle** — in `kernels/bootstrap_device_test.rs`:

- `poisson_device_draw_is_deterministic` — the same `(base state, rand_seed, n)`
  drawn twice yields **bit-identical** weight vectors.
- `poisson_device_draw_matches_host_knuth_reference` — an in-file host transcription
  of the same Knuth algorithm over the same stream produces the same weights (exact
  where the algorithm is integer-valued; otherwise ≤1e-12). This is the
  "device-vs-a-defined-in-repo-reference" criterion.
- `poisson_device_weights_are_finite_and_nonnegative` — every weight is finite and
  `>= 0.0`, and the mean is within a stated sanity band of the Poisson(1) mean over
  a large `n`.
- `poisson_is_not_reachable_from_train` — assert the session gate declines a Poisson
  config (`Ok(None)`), so the capability cannot leak into a fit.

- Run:
  `cargo test -p cb-train --test poisson_contract_test`
  `cargo test -p cb-train --no-default-features --features rocm --test poisson_contract_test`
  `cargo test -p cb-backend --no-default-features --features rocm --lib poisson -- --nocapture --test-threads 1`

### 2. Green

1. **Before changing the variant**, run
   `grep -rn "CbError::Degenerate" crates/ --include=*.rs | grep -i poisson` and
   `grep -rn "not supported on CPU" crates/ --include=*.rs` to find every matcher and
   message assertion. Update them all in the same change (R15).
2. Add the up-front rejection in `boosting.rs`.
3. Change the `bootstrap.rs` variant + message + doc.
4. Add the session-gate decline.
5. Strengthen the sampler oracle.
- Run: the three commands above.

### 3. Refactor

- Put the rejection message in one `const` (or a small `fn poisson_unsupported() ->
  CbError`) used by both `boosting.rs` and `bootstrap.rs`, so the two sites can never
  drift and the test can assert against the single source.
- Run: the three commands.

### 4. Verify

- Run: `cargo test -p cb-train --test poisson_contract_test`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test poisson_contract_test`
- Run: `cargo test -p cb-train --lib bootstrap` (the dispatch unit test)
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib bootstrap -- --test-threads 1`
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` (**blocking**)
- Run: `cargo test -p cb-train` (CPU suite; one known pre-existing red)
- Run: `cargo test -p catboost-rs` and `cargo test -p cb-model` — any crate that
  matched on the old variant must still compile and pass.
- Confirm: `grep -rn "poisson" crates/ --include=*.rs -i` shows no remaining claim
  that Poisson is "supported on GPU" by this project.

## Completion criteria

- [ ] A Poisson fit returns the SAME `CbError::Degenerate` variant and message
      under `cpu` and `rocm` builds.
- [ ] The rejection happens before any device session or gradient work.
- [ ] The device Poisson sampler is deterministic, matches an in-repo host Knuth
      reference, and is finite / non-negative.
- [ ] The session gate declines Poisson independently of the host gate.
- [ ] Every `Degenerate`-matching site for Poisson was found and updated.
- [ ] No ≤1e-5-vs-upstream claim exists anywhere for Poisson, and the intentional
      divergence from the CPU path is stated in code.

## Alternative A — if the user overrides D5 and wants a runnable device Poisson

Scoped, NOT scheduled. The minimal honest delta:

1. Add an explicit opt-in parameter (e.g. `allow_gpu_only_poisson: bool`, default
   `false`) — never enable by default, so no existing program changes behaviour.
2. Add `Poisson` to the TASK-06 allow-list **only** under that opt-in.
3. Use Design **B** for this arm alone (there is no CPU sampler to host-run), i.e.
   the existing `launch_bootstrap_weights_resident` Poisson kernel.
4. Fix `session.rs:1567-1577`: the fixed `n`-draw advance is not consumption-faithful
   for Knuth Poisson. Either make the kernel emit its consumed-draw count and advance
   the host stream by it, or move the whole stream on-device for this arm.
5. Acceptance for that arm: determinism (same seed ⇒ same model), device-vs-in-repo
   reference for the sampler, finiteness, and a model-metadata marker recording that
   the model is **not** upstream-comparable.
6. Explicitly document that no ≤1e-5 claim exists and that the model differs from
   both CatBoost CPU (which errors) and CatBoost GPU (different growing algorithm).

Estimated additional scope: one task of similar size to TASK-06, plus the stream
fidelity fix. The recommendation stands: accept D5 and keep this out of the phase.

## Risks and guardrails

- **R15 variant change breaks a caller** — the grep-first step is mandatory, and
  `catboost-rs` / `cb-model` are re-tested.
- **Reintroducing config-dependent behaviour** — the contract test is deliberately
  ungated by feature so it runs under both builds; do not add a `#[cfg]` to make it
  pass.
- **Deleting the accurate `session.rs` Poisson note** — it records a real property
  (variable draw consumption). Preserve it and cross-reference WR01-S16.
- **Over-claiming in the sampler oracle** — the module doc must say "capability, not
  parity", or a future reader will treat a green test as an upstream guarantee.

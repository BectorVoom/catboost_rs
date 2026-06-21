# Phase 8: Python Bindings, Dual API & Packaging - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-21
**Phase:** 8-python-bindings-dual-api-packaging
**Areas discussed:** Dual-API layering, Unsupported-param policy, Wheel naming & backend, Input ingest & GIL

User framing (carried forward, not re-asked): "design rust native (build pattern). python wrapper is catboost mirror."

---

## Dual-API layering

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror upstream (one hierarchy) | sklearn methods baked onto CatBoostClassifier/Regressor/Ranker, exactly like real CatBoost; no separate sklearn module | ✓ |
| Separate sklearn wrapper | Pure CatBoost-native classes + thin sklearn-adapter layer delegating to them | |

**User's choice:** Mirror upstream (one hierarchy).

### Class layer (follow-up)
| Option | Description | Selected |
|--------|-------------|----------|
| Python shim over PyO3 core | Minimal PyO3 core + hand-written pure-Python package implementing user-facing classes (mirrors upstream .pyx-over-C++) | |
| All in PyO3 (`#[pyclass]`) | Expose CatBoostClassifier/Regressor/Ranker/Pool directly as PyO3 pyclass types, no Python-level package | ✓ |

**User's choice:** All in PyO3.
**Notes:** Consequence flagged — `get_params`/`set_params`, `__sklearn_tags__`, clone, NotFittedError, large param signatures must be implemented in Rust.

### check_estimator strictness (follow-up)
| Option | Description | Selected |
|--------|-------------|----------|
| Full pass, hard gate | check_estimator passes clean, zero skips | ✓ (later revised) |
| Core checks, documented skips | Pass essential contract; documented allowlist of skipped checks | |

**User's choice:** Full pass, hard gate — **later revised** during the Input-ingest area when a direct conflict with strict dtype rejection surfaced (see that area). Net outcome: core estimator contract is a hard gate; dtype/contiguity-related checks become documented skips.

---

## Unsupported-param policy

| Option | Description | Selected |
|--------|-------------|----------|
| Reject with clear error | Typed exception naming the unsupported param and why | ✓ |
| Accept + warn, ignore | Accept known-CatBoost params, warn on ignored ones, run with subset | |
| Reject unknown, warn unsupported | Three-tier: unknown→error, known-unimplemented→warn+ignore, supported→run | |

**User's choice:** Reject with clear error.

### Validation timing (follow-up)
| Option | Description | Selected |
|--------|-------------|----------|
| At fit() time | `__init__` stores params verbatim (satisfies check_estimator); error raises in fit() | ✓ |
| At construction time | Raise immediately in `__init__` (breaks sklearn no-work-in-init; fails check_estimator) | |

**User's choice:** At fit() time.

### Param registry (follow-up)
| Option | Description | Selected |
|--------|-------------|----------|
| Full CatBoost vocab registry | Track complete CatBoost param vocabulary; distinguish known-but-unimplemented vs unrecognized/typo (with suggestions) | ✓ |
| Supported-set only | Track only implemented params; uniform "unsupported parameter" error | |

**User's choice:** Full CatBoost vocab registry.

---

## Wheel naming & backend

| Option | Description | Selected |
|--------|-------------|----------|
| Separate dist per backend | `catboost-rs` (cpu) + `catboost-rs-rocm` (rocm) as distinct PyPI projects | |
| One dist, extras/selector | Single `catboost-rs` project; backend via extras (`catboost-rs[rocm]`) | ✓ |

**User's choice:** One dist, extras/selector.
**Notes:** Flagged that PyPI extras pull Python deps, not alternate compiled binaries — exact mechanism is a research item; user intent (single discovery name, backend as install option) captured.

### Import name (follow-up)
| Option | Description | Selected |
|--------|-------------|----------|
| `catboost_rs` (own name) | `import catboost_rs`; collision-free, coexists with real catboost | ✓ |
| Shadow `catboost` | Import as `catboost` for zero-change migration; collides with real install | |
| `catboost_rs` + opt-in shim | Default own name; optional shim aliases it as `catboost` | |

**User's choice:** `catboost_rs` (own name).

---

## Input ingest & GIL

| Option | Description | Selected |
|--------|-------------|----------|
| Zero-copy where safe, else copy | Borrow contiguous correct-dtype buffers while GIL held; copy/convert other cases; own/quantize before GIL release | ✓ |
| Always copy under GIL | Copy/quantize every input into owned Rust buffers, then release GIL | |

**User's choice:** Zero-copy where safe, else copy.

### Validation strictness (follow-up)
| Option | Description | Selected |
|--------|-------------|----------|
| Coerce silently, reject only impossible | Auto-convert float64→float32, non-contig→contig, cat→codes; error only when impossible | |
| Coerce safe, warn lossy | Coerce freely, warn on result-changing conversions | |
| Strict, reject mismatches | Require correct dtype/contiguity; raise with fix-it messages | ✓ |

**User's choice:** Strict, reject mismatches.

### Conflict reconciliation (raised by Claude)
Conflict: strict float64/non-contig rejection cannot coexist with the full `check_estimator` hard gate (check_estimator feeds float64 + non-contiguous arrays).

| Option | Description | Selected |
|--------|-------------|----------|
| Coerce dtype/contiguity, strict on semantics | Convert float64/non-contig (check_estimator passes), strict only on ambiguous/unsupported | |
| Keep strict, downgrade check_estimator | Keep strict rejection; relax Area-1 to core-checks-with-documented-skips (dtype checks skipped) | ✓ |

**User's choice:** Keep strict, downgrade check_estimator. → Revises the earlier "full pass, hard gate" to "core contract hard gate + documented dtype-check skips."

---

## Claude's Discretion

- Exact PyO3 crate/module name and internal file layout.
- PYAPI-05 exception class taxonomy (names/hierarchy), provided typed thiserror variants map to specific Python exceptions with actionable messages.
- Which concrete check_estimator checks land on the documented-skip allowlist (must be enumerated/justified at implementation).

## Deferred Ideas

- Ranker → sklearn mapping (sklearn has no native ranker base).
- `.cbm` model-format interop with upstream CatBoost.
- abi3 vs free-threaded wheel build mechanics.
- Exact PyPI extras → binary-backend realization mechanism.
- `wgpu`/`cuda` wheels (only cpu + rocm required this phase).
- Shadow-`catboost` import / opt-in compat shim (rejected this phase; noted for future).

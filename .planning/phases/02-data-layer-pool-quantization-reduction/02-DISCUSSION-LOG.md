# Phase 2: Data Layer — Pool, Quantization & Reduction - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-13
**Phase:** 2-Data Layer — Pool, Quantization & Reduction
**Areas discussed:** Pool ↔ QuantizedPool design, Ingestion scope boundary, Reduction primitive, Quantized bin storage

---

## Pool ↔ QuantizedPool design

### Type relationship
| Option | Description | Selected |
|--------|-------------|----------|
| Two distinct types | `Pool` + `QuantizedPool` separate; `pool.quantize() -> QuantizedPool`; mirrors upstream raw/quantized provider split | ✓ |
| One type, internal state | Single `Pool` lazily carrying bins; blurs raw-vs-quantized boundary | |
| You decide | Defer to research | |

### Raw data ownership
| Option | Description | Selected |
|--------|-------------|----------|
| Owned now, zero-copy seam later | `Pool` owns Vec columns; ingestion trait lets a borrowed NumPy view plug in at Phase 8 | ✓ |
| Lifetime-generic / Cow from day one | `Pool<'a>` borrows immediately; lifetime complexity propagates everywhere | |
| You decide | — | |

### "Buffers reused across rounds" meaning
| Option | Description | Selected |
|--------|-------------|----------|
| Immutable bins + caller scratch | QuantizedPool immutable; trainer reuses its own histogram/gradient scratch over fixed bins | ✓ |
| QuantizedPool owns reusable scratch | QuantizedPool hands out mutable scratch; couples data layer to training allocation | |
| You decide | — | |

**User's choice:** Two distinct types · Owned now, zero-copy seam later · Immutable bins + caller scratch
**Notes:** Clean ownership story; the exact training-scratch reuse boundary is a Phase 3 concern.

---

## Ingestion scope boundary

### What Phase 2 builds (PyO3 is Phase 8)
| Option | Description | Selected |
|--------|-------------|----------|
| Rust-native ingestion + trait seam | Ingest from Rust sources behind a trait; PyO3/NumPy zero-copy wired in Phase 8 reusing the seam | ✓ |
| ndarray-only seam now | Only ndarray/slices now; Arrow/Polars + NumPy all deferred to Phase 8 | |
| You decide | — | |

### Which Rust ingestion sources wired + tested in Phase 2 (multi-select)
| Option | Description | Selected |
|--------|-------------|----------|
| ndarray / raw slices | Baseline copy-in path | (not selected — raw `Vec` remains the implicit trivial primitive) |
| Arrow (arrow-rs) | Rust arrow crate ingestion with validation | ✓ |
| Polars | Rust polars DataFrame ingestion (Arrow-backed) | ✓ |

### Where validation lives / failure mode
| Option | Description | Selected |
|--------|-------------|----------|
| At ingestion boundary, typed CbError | Validate in trait impls; typed thiserror variants Phase 8 maps to Python exceptions | ✓ |
| Defer error taxonomy to Phase 8 | Generic errors now | |
| You decide | — | |

**User's choice:** Rust-native ingestion + trait seam · Arrow + Polars · At ingestion boundary, typed CbError
**Notes:** Raw owned-`Vec` construction is treated as the implicit primitive for Builder/oracle tests; the user-facing validated external paths are Arrow + Polars.

---

## Reduction primitive

### Owner crate
| Option | Description | Selected |
|--------|-------------|----------|
| cb-core | Shared by all crates; no new dep edges | ✓ |
| cb-data | First point of need, but forces compute/train to depend on cb-data for summation | |
| You decide | — | |

### Enforcement of "only summation primitive"
| Option | Description | Selected |
|--------|-------------|----------|
| CI grep backstop + convention | Mirror the anyhow-ban: CI grep fails on raw float sums in library crates | ✓ |
| Convention + review only | Document + rely on review | |
| You decide | — | |

### Which C++ accumulation behavior to match
| Option | Description | Selected |
|--------|-------------|----------|
| Research must pin it exactly | Researcher determines f64 accumulator + exact order (naive/Kahan/pairwise) | ✓ |
| Assume naive f64 sequential | Default left-to-right; revisit on divergence | |
| You decide | — | |

**User's choice:** cb-core · CI grep backstop + convention · Research must pin it exactly
**Notes:** Structural enforcement, consistent with Phase 1's structural anyhow ban.

---

## Quantized bin storage

### Bin width selection / u32 support
| Option | Description | Selected |
|--------|-------------|----------|
| u8 ≤255 borders else u16; no u32 | Matches DATA-02 literal text | |
| u8/u16/u32, match upstream exactly | Also support u32 high-cardinality; broader parity surface | ✓ |
| You decide | — | |

### Column exposure to Phase 3 trainer
| Option | Description | Selected |
|--------|-------------|----------|
| Typed per-column enum (U8/U16[/U32]) | enum `{ U8, U16, U32 }`; kernel matches on width; zero widening cost | ✓ |
| Uniform widen-to-u16 view | One kernel path; loses u8 memory/bandwidth win | |
| You decide | — | |

### SoA layout
| Option | Description | Selected |
|--------|-------------|----------|
| Per-feature buffers (Vec per column) | Each column its own contiguous buffer; cache-friendly per-feature passes | ✓ |
| Single arena + offsets | One allocation + offset slices; mixed widths complicate the arena | |
| You decide | — | |

**User's choice:** u8/u16/u32 match upstream exactly · Typed per-column enum (extended to U8/U16/U32) · Per-feature Vec buffers
**Notes:** Deliberate one-width-beyond-the-requirement choice for exact parity; the typed enum gains a `U32` arm accordingly.

---

## Claude's Discretion

- GreedyLogSum border algorithm, `<`/`<=` assignment, NaN/duplicate-column handling — parity-dictated.
- `NanMode` (Min/Max/Forbidden) and categorical hash function — match upstream exactly.
- Auto class-weight formulas (Balanced/SqrtBalanced) and per-object/per-class weights — parity-dictated.
- Intermediate-oracle fixture schema for borders/quantization — research/planning.
- Whether Polars rides the Arrow code path or is separate.
- Concrete `arrow`/`polars` crate versions (latest stable).

## Deferred Ideas

- PyO3 / NumPy zero-copy ingestion → Phase 8 (reuses the ingestion trait seam).
- CTR / ordered target-statistic computation → Phase 5 (Pool stores raw categoricals + hashing only).
- GPU bin storage / kernels → Phase 7 (QuantizedPool stays CubeCL-free).

# Pitfalls Research

**Domain:** Full Rust rewrite of a numerically-exact gradient boosting library (CatBoost), with multi-backend GPU (CubeCL) and PyO3 bindings, oracle-tested to ≤1e-5 against the C++ reference.
**Researched:** 2026-06-13
**Confidence:** HIGH for numerical-parity / algorithmic-fidelity / Rust-ergonomics pitfalls (grounded in the vendored CatBoost source + established float-reproducibility literature); MEDIUM for CubeCL/ROCm specifics (rapidly-moving, WIP backend) and PyO3 free-threaded ABI (evolving with Python 3.13t/3.15).

> Scope note: this is the single most failure-prone class of project in ML engineering — a *bit-reproducing rewrite of a mature, heavily-optimized C++ trainer in a different language, plus a different GPU framework, plus a different binding layer.* The dominant risk is not "will it work" but "will it match," and parity failures are diffuse, late-surfacing, and expensive to localize. Phase the work so that **a self-consistent CPU oracle exists before any GPU or binding work begins.**

---

## Critical Pitfalls

### Pitfall 1: Treating ≤1e-5 parity as a single end-of-project gate instead of a per-stage invariant

**What goes wrong:**
The team builds quantization → tree-building → leaf estimation → prediction → categorical encoding, then turns on the oracle harness at the end. End-to-end predictions are off by 1e-2 or worse, and there is no way to tell *which* stage diverged. Because gradient boosting is sequential, a tiny divergence in tree 1 (e.g. one border placed differently) compounds across 1000 trees into a large prediction gap. You end up bisecting a 1000-tree, multi-permutation pipeline by hand.

**Why it happens:**
The oracle in PROJECT.md is defined only on the *final output* (predictions within 1e-5). Teams naturally test what the spec measures. But final-output parity is the *hardest* thing to achieve and the *least* diagnostic when it fails.

**How to avoid:**
Define intermediate oracles at every stage boundary, exported from the C++ reference: (a) quantization borders per feature, (b) quantized bin indices per object, (c) the chosen split (feature+border) at each tree depth, (d) leaf values per tree, (e) raw approx after each tree, (f) final prediction. Build a debug build of CatBoost that can dump these (it has internal logging and `--detailed-profile`; you can also instrument it). Match each stage independently before composing. A first-tree, single-permutation, no-categorical, regression model must match *bit-for-bit on the split decisions* before anything else proceeds.

**Warning signs:**
Predictions match for tree count 1 but drift as trees increase; parity passes for one dataset but not another; you can only describe failures as "the number is wrong" not "border 7 of feature 3 differs."

**Phase to address:**
Phase 0/1 (oracle harness + quantization). The intermediate-oracle infrastructure must exist *before* tree-building is written.

---

### Pitfall 2: Floating-point summation order divergence (the #1 parity killer)

**What goes wrong:**
Histogram accumulation, gradient/hessian sums per leaf, and score computation all sum many floats. IEEE float addition is **not associative**: `(a+b)+c ≠ a+(b+c)` at the ULP level. CatBoost's C++ sums in a specific order (often blocked, sometimes SIMD-vectorized, sometimes Kahan/`TKahanAccumulator`, sometimes thread-partitioned then merged). If your Rust sums in a different order — e.g. you use `iter().sum()`, or Rayon partitions the work differently, or you fold left where C++ folds in pairs — leaf values differ in the last few ULPs. Over 1000 boosting rounds this compounds past 1e-5.

**Why it happens:**
Rust idiom (`.iter().sum()`, `.fold()`, Rayon `par_iter().sum()`) hides summation order, and the order differs from C++. Parallel reductions are *especially* dangerous: the reduction-tree shape depends on thread count and chunk size, so results vary run-to-run and machine-to-machine even within Rust. This is the textbook cause of irreproducibility in parallel numeric code.

**How to avoid:**
- Read the *exact* accumulation strategy from the C++ source for each reduction (look at `approx_calcer.cpp`, `scoring.cpp`, `leafwise_scoring.cpp`, and the `TKahanAccumulator` usages) and replicate the *same blocking and the same accumulator type* (plain `double` vs Kahan-compensated).
- CatBoost accumulates many statistics in `double` even for `float` inputs — match the accumulator width exactly; an `f32` accumulator where C++ uses `double` will not paritymatch.
- For parallel sums, make the partitioning deterministic: fixed chunk sizes, fixed thread count in the oracle harness, deterministic reduction order (sum partial results in a fixed index order, not in completion order). Do **not** use a parallel `sum()` that merges in nondeterministic order.
- Forbid `-ffast-math`-equivalent behavior. In Rust this means **never use `f32::mul_add`/FMA where C++ used separate mul+add (or vice versa)** — FMA changes rounding. Avoid `std::intrinsics::fadd_fast` / the `fast-float` style. Keep `target-feature` consistent and avoid auto-FMA-contraction surprises (Rust does not contract by default, which is good, but verify).

**Warning signs:**
Parity passes single-threaded but fails multi-threaded; parity differs between x86 and ARM, or between two machines; the diff is always in the last 2–4 decimal digits and grows with tree count; results change when you change Rayon thread-pool size.

**Phase to address:**
Phase 1 (quantization sums) and Phase 2 (tree building / leaf estimation). Establish a "summation discipline" rule early and a single audited reduction utility used everywhere.

---

### Pitfall 3: Quantization / border-selection algorithm mismatch

**What goes wrong:**
CatBoost offers multiple border-selection algorithms (confirmed in `grid_creator.cpp`: `GreedyLogSum` — the CPU default — plus `GreedyMinEntropy`, `MinEntropy`, `MaxLogSum`, `Median`, `Uniform`, `UniformAndQuantiles`). Each produces a different set of split borders, which changes *every* downstream split decision. If your border computation differs even slightly — tie-breaking when two values are equal, handling of `NaN` (`NanMode`: Min/Max/Forbidden), per-feature border count caps, or how quantiles are interpolated — every tree built afterward diverges, and final predictions miss 1e-5 by orders of magnitude.

**Why it happens:**
Quantization looks like "just bin the floats," but it is a precise greedy optimization with subtle tie-breaking and NaN/inf placement rules. The default `GreedyLogSum` is a weighted greedy split that is easy to approximate but hard to reproduce exactly. Edge cases (all-equal feature, fewer distinct values than requested borders, NaN handling) are where divergence hides.

**How to avoid:**
- Port `GreedyLogSum` first and **only** that algorithm for v1; oracle-test the border *set* per feature against the C++ output before doing anything downstream. Treat borders as a hard intermediate oracle (Pitfall 1).
- Reproduce NaN handling exactly (`NanMode` default is `Min` for most, but verify per feature type) — NaN-as-a-bin is a common silent mismatch.
- Match the per-feature `border_count` default (254 for CPU) and the behavior when distinct values < border_count.
- Match float→bin assignment boundary semantics exactly (`<` vs `<=` at the border). One off-by-one in `value < border` vs `value <= border` shifts an entire column of bins.

**Warning signs:**
Borders match for "nice" synthetic data but not for data with duplicates, NaNs, or near-equal values; the number of bins differs by one; bin assignment differs only for values exactly equal to a border.

**Phase to address:**
Phase 1 (quantization). This is the *foundation* — nothing downstream can parity-match until borders do.

---

### Pitfall 4: Ordered boosting and ordered target statistics implemented subtly wrong

**What goes wrong:**
Ordered boosting (CatBoost's defining feature) maintains multiple random **permutations** of the data and, for each object, computes its target statistic / gradient using only objects that come *before* it in the permutation — preventing target leakage. The implementation is full of off-by-one and "prefix" subtleties: which objects are "before," how the running prefix sums (`sumCount`, `sumTarget`) are maintained per permutation, the prior used for the very first objects, how categorical target stats use the *same* permutation, and the interaction with the `fold`/`TFold` structures (confirmed: `fold.cpp`, `learn_context.cpp` seed the RNG and build folds). Get the prefix boundary wrong by one and you reintroduce leakage — the model still trains and predicts plausibly, so tests pass but parity fails and results are subtly *better* on train (a tell-tale leakage signature).

**Why it happens:**
The ordered-boosting prefix logic is the least-documented, most-intricate part of the algorithm. The "use only the prefix" rule has edge cases (first object, ties, the prior term `(sumTarget + prior) / (sumCount + priorWeight)`), and the permutation count, permutation block structure, and seed all feed in. It is easy to write something that looks correct and is leakage-free *enough* to train, but not bit-identical.

**How to avoid:**
- Implement *plain* (non-ordered) boosting first and oracle-match it; only then add ordered mode. This isolates the permutation logic.
- Replicate the exact prior formula and the exact prefix boundary (strictly-before vs up-to-and-including) by reading `approx_calcer.cpp` and the categorical target-statistic code (`private/libs/algo/online_ctr.*` / ctr computation) line by line.
- Add an intermediate oracle on the *target statistic value per object per permutation*, not just final predictions.
- Reproduce the permutation generation exactly (see Pitfall 5) — ordered boosting is meaningless if the permutation differs.

**Warning signs:**
Train metrics are suspiciously good (leakage), test metrics drift; parity fails only when `boosting_type=Ordered` (the default for small datasets) but passes for `Plain`; the gap appears in early trees where the prefix is short.

**Phase to address:**
Phase 2–3 (boosting core + categorical features). Sequence: plain boosting → ordered boosting → ordered target statistics.

---

### Pitfall 5: RNG / seed non-reproducibility across the language boundary

**What goes wrong:**
CatBoost uses a specific PRNG (`TFastRng64` / its `TRandom` wrapper, seeded by `Params.RandomSeed`, confirmed at `learn_context.cpp:467 , Rand(randomSeed)`) to generate the data permutations, MVS/bootstrap sampling, random feature subsampling (`rsm`), and `random_strength` score perturbation (`rand_score.cpp`). If your Rust uses a *different* PRNG (e.g. `rand`'s `StdRng`/ChaCha, or `SmallRng`), the permutations and sampling differ, so ordered boosting and every stochastic step diverge — guaranteeing parity failure even when the math is perfect.

**Why it happens:**
"It's just randomness, any RNG seeded the same is fine" — but parity requires the *same bit sequence*, which means the *same PRNG algorithm*, the *same seeding/warm-up*, the *same draw order*, and the *same consumption pattern* (how many draws per permutation, whether floats or ints are drawn, the float-from-bits conversion).

**How to avoid:**
- Port CatBoost's exact PRNG (`TFastRng64`, a SplitMix/xorshift-family generator) into Rust as a dedicated, oracle-tested module *before* any stochastic algorithm. Oracle-test the raw bit stream for a fixed seed against the C++ generator.
- Match the *order and granularity* of draws: permutation shuffles (Fisher-Yates direction and modulo-vs-multiply index selection), `random_strength` perturbation draws, MVS sampling draws. A different shuffle direction produces a different permutation from the identical bit stream.
- Make seed handling explicit end-to-end and ensure the Python binding passes `random_seed` through unchanged.

**Warning signs:**
Parity fails for any non-zero `random_strength`, any subsampling, or ordered mode, but passes for fully deterministic configs; changing the PRNG implementation changes results; permutations differ for the same seed.

**Phase to address:**
Phase 1 (a foundational, oracle-tested RNG module), consumed by Phase 2/3.

---

### Pitfall 6: CPU vs GPU result divergence beyond tolerance (CubeCL)

**What goes wrong:**
The GPU training path (histogram construction, score computation, leaf estimation) produces predictions that differ from the CPU path by more than 1e-5 — and from the C++ oracle by more. GPU divergence is *expected* in float reductions (different parallel reduction order, atomics ordering, FMA contraction in GPU codegen, lower-precision intermediate accumulation). Worse, CatBoost's *own* GPU and CPU implementations are not bit-identical to each other (the C++ has separate `catboost/cuda/` algorithms), so "match GPU to the C++ GPU oracle" and "match GPU to your CPU path" are two *different, possibly incompatible* targets.

**Why it happens:**
GPU reductions use atomics and warp/wavefront-level operations whose ordering is nondeterministic; GPU compilers contract `a*b+c` into FMA aggressively; histogram atomic-add order is data-race-determined. ROCm/HIP adds a specific trap: **wavefront size is 64 on AMD vs 32-thread warps on NVIDIA** — warp-synchronous kernel code that assumes 32 lanes compiles and runs on AMD but produces *wrong results*. CubeCL's ROCm/HIP backend is explicitly **work-in-progress** with raw, unimproved bindgen bindings.

**How to avoid:**
- Decide *up front* whether the GPU oracle is "match C++ CPU within 1e-5" or "match C++ GPU." Given PROJECT.md tests `rocm` only and the 1e-5 bar, the realistic target is: **CPU path is the source of truth and must hit 1e-5 vs C++; GPU path must hit a (looser, separately-justified) tolerance vs the Rust CPU path.** Do not promise 1e-5 GPU-vs-C++-CPU parity — get explicit sign-off on a GPU tolerance.
- Never write warp-size-assuming kernels; use CubeCL's plane/subgroup abstractions and never hardcode 32 or 64. Test on `rocm` (wavefront 64) early, not just `wgpu`/`cpu`.
- Make GPU reductions as deterministic as feasible (fixed tile sizes, tree-reduction instead of atomic-add where exactness matters, avoid `mul_add` contraction surprises).
- Defer GPU entirely until the CPU path is fully oracle-passing. GPU is a *parallel* implementation of an *already-correct* algorithm, never the place to discover the algorithm.

**Warning signs:**
GPU results match `wgpu`/`cpu` backends but diverge on `rocm`; divergence scales with object count (reduction-order effect); kernels that read neighboring-lane values give garbage on AMD; results change run-to-run on GPU (atomic nondeterminism).

**Phase to address:**
A *late* phase, after CPU parity is locked. Flag for deep research before starting (CubeCL maturity + ROCm wavefront semantics).

---

### Pitfall 7: PyO3 zero-copy NumPy/Arrow buffers vs Rust lifetime/GIL safety

**What goes wrong:**
To meet the memory-efficiency goal, the binding borrows NumPy/Arrow buffers zero-copy. But a borrowed `&[f32]` from a NumPy array is only valid while the GIL is held and the Python object is alive. If Rust holds the slice across a `Python::allow_threads` (GIL release) boundary, or stores it beyond the call, Python's GC can free or resize the buffer → use-after-free. With the free-threaded (no-GIL) Python builds now landing in 3.13t/3.14+, the implicit "GIL protects the buffer" assumption disappears entirely. Mismatched dtype/strides (non-contiguous arrays, `float64` vs `float32`, Fortran order) silently mis-read the buffer.

**Why it happens:**
`numpy`/`PyArray` give an ergonomic `as_slice()` that *looks* like an owned borrow, and under the GIL it "works." Developers release the GIL during the long training loop (correct for performance), forgetting the borrowed buffers are now unprotected. Arrow zero-copy adds its own ownership/lifetime rules (the C Data Interface release callback).

**How to avoid:**
- For training (long-running), **copy/quantize input into Rust-owned columnar storage up front** while the GIL is held, then release the GIL. Zero-copy only for short prediction calls where the GIL is held the whole time.
- For Arrow, use the C Data Interface correctly and hold an owning handle (do not detach the release callback); validate buffer contiguity and dtype before borrowing.
- Validate dtype, contiguity, and shape explicitly; reject or copy non-contiguous / wrong-dtype arrays rather than reinterpreting bytes.
- Be free-threaded-aware: do not rely on the GIL for buffer protection; design for `Py_GIL_DISABLED` from the start.

**Warning signs:**
Segfaults or corrupted predictions only under load / with the GC active; works for small arrays but crashes for large ones (reallocation); wrong predictions for `float64` input or sliced/transposed DataFrames; crashes only on free-threaded interpreters.

**Phase to address:**
Phase (Python bindings), but the "own your columnar data" decision must be made in the core data-layout phase (Phase 1) so the boundary can copy-in cheaply.

---

### Pitfall 8: Per-backend wheel / ABI distribution complexity (maturin, Python ≥3.12)

**What goes wrong:**
The plan ships per-backend wheels (`catboost-rs-rocm`, etc.). Each backend × Python-version × OS × CPU-arch is a separate build, and the wheel matrix explodes. Using `abi3` (stable ABI) to collapse the Python-version axis seems attractive — but **`abi3` is not available for free-threaded builds**; you cannot build a single `abi3` wheel that covers both GIL and free-threaded 3.13t/3.14. The newer `abi3t` covers free-threaded but only on Python **3.15+**. So Python 3.12–3.14 + free-threaded requires version-specific wheels anyway. GPU backends also can't really use `abi3` cleanly if they link backend-specific native libs.

**Why it happens:**
The wheel/ABI story for the 3.12→3.15 transition with free-threading is genuinely in flux (PEP 703/779/803), and PyO3/maturin support is still landing. Teams assume "abi3 = one wheel per platform" and discover the free-threaded exception late.

**How to avoid:**
- Decide the support matrix explicitly and minimally: pick a small set of (Python versions × OS × backend) for v1; do not promise free-threaded support in v1 unless required.
- Use `maturin-action` + a generated CI matrix from day one; treat the wheel matrix as code, not a manual step.
- Default to version-specific (non-abi3) wheels initially — simplest correct path — and only adopt `abi3`/`abi3t` once the support story stabilizes and PyO3/maturin support is confirmed for your target versions.
- Keep the GPU-backend native dependency (ROCm/CUDA runtime) clearly documented as a *system* prerequisite per wheel; do not attempt to bundle the GPU runtime.

**Warning signs:**
Wheels build locally but fail on free-threaded interpreters; `abi3` build errors with `Py_GIL_DISABLED` defined; import errors due to missing system ROCm/CUDA libs; combinatorial CI that takes hours.

**Phase to address:**
Phase (Python packaging), but scope the matrix at roadmap time. Flag for research: confirm current PyO3/maturin abi3t status before committing to an ABI strategy.

---

### Pitfall 9: Borrow-checker friction with mutable columnar state in the hot loop

**What goes wrong:**
Gradient boosting's hot loop mutates shared columnar buffers (per-leaf gradient/hessian histograms, per-object approx vectors, fold prefix sums) while iterating, and wants multiple concurrent mutable views into disjoint slices of the same arrays (per-leaf, per-permutation). The borrow checker forbids overlapping `&mut`, so the naive fix is `.clone()` the buffers per iteration — directly violating the memory-efficiency constraint and tanking performance. Alternatively teams reach for `RefCell`/`Rc<RefCell<...>>` everywhere (runtime borrow panics, which also violates no-`unwrap`/no-panic discipline) or `unsafe` pointer aliasing (UB risk).

**Why it happens:**
The C++ reference freely aliases mutable arrays via raw pointers and indices. Translating that idiom directly fights Rust. The memory-efficiency goal makes `.clone()` unacceptable, pushing developers toward `unsafe` or interior mutability before they understand the safe options.

**How to avoid:**
- Use `split_at_mut` / `chunks_mut` / `iter_mut` to get *disjoint* mutable slices (per-leaf partitions are naturally disjoint) — this is the idiomatic, safe, zero-copy answer for most of the hot loop.
- Use index-based access into a single owned arena/`Vec` (Struct-of-Arrays columnar layout) rather than holding many references; pass indices, not `&mut`.
- For genuinely-needed parallel disjoint mutation, use Rayon's `par_chunks_mut` (safe, deterministic if chunking is fixed — ties into Pitfall 2).
- Reserve `unsafe` for a *tiny*, audited, separately-tested core (e.g. an unchecked-index histogram kernel) with a safe wrapper and a documented invariant — never sprinkle it.

**Warning signs:**
`.clone()` calls appearing in profiling as top allocations; `RefCell` borrow panics (which also break the no-panic rule); growing `unsafe` blocks without safety comments; profiler shows allocator pressure inside the per-tree loop.

**Phase to address:**
Phase 1 (data layout: commit to SoA columnar arena early) and Phase 2 (tree-building hot loop).

---

### Pitfall 10: Oracle harness — getting C++ CatBoost into CI deterministically

**What goes wrong:**
The oracle requires running the original CatBoost to generate reference outputs. But building CatBoost from `catboost-master/` takes 10–60 min (462+ C++ files via CMake, needs Python + libclang + correct LLVM), per CONCERNS.md — so CI either is brutally slow or skips the oracle, defeating the entire parity strategy. Worse, if reference outputs are regenerated on each run, *CatBoost's own nondeterminism* (thread-count-dependent float sums) makes the "expected" values flaky, so the same Rust output passes one day and fails the next.

**Why it happens:**
Building the reference is heavyweight (documented in CONCERNS.md). And regenerating reference values live assumes CatBoost is deterministic — it largely is *if* you pin `thread_count=1`, `random_seed`, and config, but teams forget and get flaky 1e-5 comparisons from the reference side.

**How to avoid:**
- **Pre-generate reference outputs as committed/cached fixtures**, not live in CI. Generate them once (pinned CatBoost version, `thread_count=1`, fixed `random_seed`, fixed params, fixed input seed), commit the expected vectors (or cache by content hash). The CI job only runs the Rust side against frozen fixtures.
- Pin the exact CatBoost version/commit used to generate fixtures and record it; re-vendoring CatBoost invalidates fixtures (tie to the version-pinning concern in CONCERNS.md).
- Force determinism on the reference: `thread_count=1`, fixed seed, single-permutation configs for the first oracles, `boosting_type` pinned.
- Compare with an explicit absolute tolerance of 1e-5 *and* track the *max* observed diff over time (regression budget) so silent drift toward the limit is visible.
- Provide a separate, opt-in "rebuild reference" job (slow, manual) decoupled from PR CI.

**Warning signs:**
CI flakiness where the *same* commit passes/fails; oracle job dominates CI wall-clock; "expected" values change when the CI runner's core count changes; nobody can say which CatBoost commit produced the fixtures.

**Phase to address:**
Phase 0 (harness/infra) — must exist before tree-building. Highest-leverage early investment.

---

### Pitfall 11: Scope risk — full parity as a monolith that stalls

**What goes wrong:**
"Full feature parity as v1" (categorical + text + embeddings + SHAP + ranking + all loss functions + cross-version serialization + 4 GPU backends + dual Python API) is enormous. A team that tries to land it as one integrated push has nothing oracle-passing for months, can't tell what works, and stalls. Text/embedding features and SHAP each rival the core trainer in complexity.

**Why it happens:**
The spec lists everything as "Active / v1," and parity makes partial features feel useless ("a booster that can't do categorical isn't CatBoost"). So everything is built simultaneously and nothing finishes.

**How to avoid:**
- Phase by *oracle-passing vertical slices*, narrowest-first: (1) numeric-only regression, plain boosting, CPU, oblivious trees, single permutation, predict — oracle-passing end-to-end. Then widen one axis at a time: (2) classification/ranking losses, (3) ordered boosting, (4) categorical/ordered target stats, (5) text, (6) embeddings, (7) SHAP, (8) serialization, (9) GPU, (10) Python bindings.
- Each slice must hit 1e-5 before the next starts — a passing narrow slice is worth more than ten half-built features.
- Symmetric (oblivious) trees only for v1 core (CatBoost's default); defer non-symmetric tree modes.
- Treat text/embeddings/SHAP/GPU as *separate milestones*, explicitly de-risked, not as part of the first parity milestone.

**Warning signs:**
Many in-progress features, none oracle-passing; "it'll all come together at the end"; inability to demo a single working configuration; roadmap phases defined by component (binding, GPU) rather than by oracle-passing capability.

**Phase to address:**
Roadmap structure itself — this pitfall is *prevented by how phases are drawn*, so it belongs in roadmap design, not a single phase.

---

### Pitfall 12: `no-unwrap` / `thiserror`-vs-`anyhow` boundary mistakes

**What goes wrong:**
The constraint bans `unwrap()` in production, but the hot numeric loop is full of indexing (`vec[i]`), slice ops, and conversions that *panic* implicitly (out-of-bounds, `usize`↔`f64` casts, `as` truncation, `slice[a..b]`). "No `unwrap`" gets satisfied superficially while panics remain everywhere. Separately, `anyhow` (an application-level type) leaks into the library/`thiserror` layer (or vice versa), so the PyO3 boundary can't translate errors into clean Python exceptions, and library consumers get opaque `anyhow::Error` instead of typed errors.

**Why it happens:**
Indexing panics don't *look* like `unwrap`, so they slip the rule. And `anyhow` is so ergonomic that it creeps into library code; the thiserror/anyhow split is a discipline, not enforced by the compiler.

**How to avoid:**
- Enforce with lints, not vigilance: `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, clippy::panic, clippy::as_conversions)]` (or a curated subset) in library crates. Indexing-slicing lint is what actually catches the hidden panics.
- Use `get()`/`get_mut()` returning `Option`, checked conversions (`TryFrom`, `f64::from`), and the audited `unsafe` get-unchecked only inside the tiny hot-kernel module (Pitfall 9) with a proven invariant.
- Rule: **`thiserror` typed errors in every library crate; `anyhow` only in the binding/CLI/test layer.** The PyO3 layer maps the library's typed error enum to Python exception types. Add a clippy/grep CI check that `anyhow` does not appear in core crates' non-test code.

**Warning signs:**
Panics in production despite "no unwrap"; `anyhow` in `Cargo.toml` `[dependencies]` of a core crate; PyO3 layer returning generic `PyRuntimeError` for everything; `as` casts silently truncating large indices.

**Phase to address:**
Phase 0 (workspace setup: lint config + error-type architecture as the very first commit, so the discipline is enforced from line 1).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use `.iter().sum()` / Rayon `par_iter().sum()` instead of replicating C++ summation order | Fast to write, idiomatic | Guaranteed 1e-5 parity failure that compounds; nondeterministic across machines | **Never** in any reduction feeding tree decisions or leaf values |
| `f32` accumulators where C++ uses `double` | Less memory | Parity failure; precision loss | Never (memory saving is negligible vs. parity loss) |
| `.clone()` columnar buffers to dodge the borrow checker | Compiles immediately | Violates memory-efficiency constraint; allocator-bound hot loop | Throwaway spike only; never in committed hot-loop code |
| Approximate `GreedyLogSum` borders | Avoids exact greedy port | Every downstream tree diverges; total parity failure | Never — borders are foundational |
| Live-regenerate oracle reference values in CI | No fixture management | Flaky CI from CatBoost's own nondeterminism; slow CI | Only with `thread_count=1` + pinned seed/version, and even then prefer cached fixtures |
| Build all parity features at once | "Looks comprehensive" | Months with nothing oracle-passing; stall | Never — vertical slices only |
| `RefCell`/`Rc<RefCell>` for mutable columnar state | Sidesteps borrow checker | Runtime borrow panics (breaks no-panic rule); overhead | Only outside the hot loop, for genuinely shared config |
| Skip ROCm wavefront-size correctness, develop on `wgpu`/`cpu` only | Faster GPU iteration | Wrong results on the one backend that's actually tested (`rocm`) | Never — `rocm` is the GPU test target |
| Zero-copy borrow NumPy across GIL release | Saves a copy | Use-after-free / corruption under GC or free-threading | Only for GIL-held short prediction calls; never across `allow_threads` for training |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| C++ CatBoost oracle | Regenerating reference values live with default (multi-thread) config | Pre-generate frozen fixtures with `thread_count=1`, pinned seed + version; commit/cache them |
| CubeCL `rocm`/HIP | Hardcoding 32-lane warp assumptions | Use plane/subgroup abstractions; test wavefront-64 (AMD) early; never assume lane count |
| CubeCL backends | Assuming `cpu`/`wgpu` parity implies `rocm` parity | Validate on `rocm` specifically; reductions diverge per backend |
| NumPy via PyO3/`numpy` crate | Borrowing `as_slice()` across GIL release; ignoring dtype/strides | Copy-in to Rust-owned SoA under GIL; validate contiguity + dtype; reject non-contiguous |
| Arrow/Polars (C Data Interface) | Detaching the release callback; assuming contiguous | Hold owning handle; honor release callback; validate layout |
| maturin / abi3 | Expecting one `abi3` wheel to cover free-threaded builds | `abi3` excludes free-threaded; use version-specific wheels until `abi3t` (3.15+) is viable |
| `bindgen` for any C glue | Pinning old `bindgen` (`~0.59`) with removed APIs (`size_t_is_usize`) | Use current bindgen API; pre-generate + commit `bindings.rs` |
| Seed/`random_seed` through Python | Python binding silently remaps or drops the seed | Pass `random_seed` byte-identical end-to-end; oracle-test the PRNG stream |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Allocating temporary `Vec`s inside the per-tree / per-leaf loop | Allocator dominates profile; poor scaling with tree count | Pre-allocate reusable scratch buffers in the fold/context struct; reuse across rounds | Any non-trivial tree count (>100) |
| Per-prediction pointer-array allocation (mirrors upstream CONCERNS.md `Vec<*const f32>`) | Throughput capped by allocator in batch inference | Pass pre-allocated buffers; SoA layout; stack arrays for small batches | High-QPS inference (>100k/s) |
| Naive parallel reduction with dynamic chunking | Nondeterministic results *and* false sharing | Fixed chunk sizes, fixed thread count, deterministic merge order | Multi-core + parity simultaneously |
| Row-major (AoS) feature storage | Cache-thrashing histogram builds; SIMD-unfriendly | Struct-of-Arrays columnar layout from Phase 1 | Wide datasets / many features |
| Recomputing quantization or CTRs per round instead of caching | Quadratic-ish training time | Cache quantized data + online-CTR prefix sums (mirror C++ `calc_score_cache`) | Large datasets / many rounds |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Borrowing Python/NumPy buffers beyond GIL-held scope | Use-after-free, memory corruption, exploitable | Copy-in to Rust-owned storage before releasing GIL; never store borrowed slices |
| `unsafe impl Send/Sync` on handles without verifying thread-safety (per CONCERNS.md) | Data races, UB under concurrent use | Verify thread-safety from source; default to `!Sync` or `Arc<Mutex<>>` until proven |
| Unchecked `unsafe` indexing in hot kernels | OOB read/write, UB | Confine to one audited module with proven invariants + safe wrapper + tests |
| Deserializing untrusted `.cbm`/model files without bounds validation | Malformed-model crash / OOB during load | Validate all offsets/sizes during deserialization; never trust embedded lengths |
| `as` truncation of large indices (`usize as u32`) | Silent wrong-bin / corruption on large data | `TryFrom` with explicit error; lint `as_conversions` |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Diverging from CatBoost parameter names/defaults in the "native" API | Drop-in migration silently changes results | Match CatBoost parameter names *and* defaults exactly; oracle-test default config |
| sklearn API not honoring sklearn contracts (`get_params`/`set_params`, `clone`, `check_estimator`) | Breaks inside `Pipeline`/`GridSearchCV` | Run sklearn's `check_estimator` as a test |
| Opaque errors at the Python boundary | Users can't diagnose bad input | Map typed `thiserror` errors to specific Python exceptions with actionable messages |
| Silent dtype coercion of `float64`→`float32` input | Subtle accuracy changes vs CatBoost | Match CatBoost's input dtype handling; warn/document explicitly |
| Requiring users to know which backend wheel to install with no guidance | Wrong-wheel import failures | Clear per-wheel system-requirement docs (ROCm/CUDA runtime versions) |

## "Looks Done But Isn't" Checklist

- [ ] **Quantization:** Often missing exact NaN/inf bin placement and border tie-breaking — verify against per-feature border-set oracle including duplicate/NaN columns.
- [ ] **Ordered boosting:** Often missing the exact prefix boundary and prior formula — verify per-object target-statistic oracle, not just final predictions; check train metrics aren't suspiciously leakage-good.
- [ ] **Summation:** Often missing C++ accumulator *width* (double) and *order* — verify multi-threaded result equals single-threaded result equals C++ to 1e-5.
- [ ] **RNG:** Often missing exact draw order/shuffle direction — verify raw PRNG bitstream matches C++ for a fixed seed before trusting any stochastic step.
- [ ] **GPU:** Often "works" only on `cpu`/`wgpu` — verify on `rocm` (wavefront 64) and against a *stated* tolerance, not assumed 1e-5.
- [ ] **PyO3 zero-copy:** Often crashes only under GC/load/free-threading — verify with large arrays, non-contiguous input, and a free-threaded interpreter.
- [ ] **No-unwrap:** Often satisfied superficially while indexing panics remain — verify `clippy::indexing_slicing` + `unwrap_used` deny passes.
- [ ] **Oracle CI:** Often flaky from reference-side nondeterminism — verify fixtures are frozen with `thread_count=1` + pinned version, and CI doesn't rebuild CatBoost.
- [ ] **Serialization:** Often round-trips your own format but not cross-version — verify load of a reference `.cbm` produced by upstream CatBoost.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Summation-order divergence found late | HIGH | Introduce one audited reduction utility matching C++; replace all ad-hoc sums; re-verify per-stage oracles bottom-up |
| Quantization borders wrong | HIGH | Borders are foundational — fix first, then every downstream oracle must be re-validated; nothing above it can be trusted until borders match |
| Ordered-boosting leakage bug | MEDIUM-HIGH | Bisect with per-permutation target-stat oracle; fix prefix boundary; re-run plain-then-ordered comparison |
| RNG mismatch | MEDIUM | Replace PRNG with exact C++ port; re-verify bitstream; re-run all stochastic-config oracles |
| GPU divergence | MEDIUM | Fall back to CPU path (still correct); treat GPU as perf-opt with explicit looser tolerance; fix determinism per-kernel |
| PyO3 use-after-free | MEDIUM | Switch hot path to copy-in ownership; audit all borrowed-buffer lifetimes; add free-threaded test |
| Monolithic stall | HIGH | Re-cut roadmap into oracle-passing vertical slices; freeze a narrow working slice; widen one axis at a time |
| Hidden panics / error-boundary leak | LOW-MEDIUM | Turn on clippy denies; convert indexing to `get()`; relocate `anyhow` out of core crates |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. Parity as end-gate not per-stage | Phase 0 (oracle infra) | Intermediate oracles exist for borders, splits, leaf values before tree-building |
| 2. Summation-order divergence | Phase 1–2 | Multi-thread == single-thread == C++ within 1e-5; one audited reduction utility |
| 3. Quantization/border mismatch | Phase 1 | Per-feature border-set oracle passes incl. NaN/dup columns |
| 4. Ordered boosting subtly wrong | Phase 2–3 | Plain matches, then ordered matches; per-object target-stat oracle |
| 5. RNG non-reproducibility | Phase 1 | Raw PRNG bitstream matches C++ for fixed seed |
| 6. CPU/GPU divergence (CubeCL/ROCm) | Late GPU phase (research-flagged) | `rocm` results within a *stated* tolerance of Rust CPU path |
| 7. PyO3 zero-copy lifetime/GIL | Python-bindings phase (layout decided Phase 1) | No UAF under GC/load/free-threading; dtype+contiguity validated |
| 8. Per-backend wheel/ABI complexity | Packaging phase (matrix scoped at roadmap) | CI matrix builds; documented system deps; ABI strategy confirmed |
| 9. Borrow-checker / clone friction | Phase 1 (layout) + Phase 2 (hot loop) | No `.clone()`/`RefCell` in hot loop profile; `split_at_mut` used |
| 10. Oracle harness in CI | Phase 0 | Frozen fixtures, pinned version, no live CatBoost rebuild in PR CI |
| 11. Scope/monolith stall | Roadmap structure | Phases defined as oracle-passing vertical slices, narrowest first |
| 12. no-unwrap / error boundary | Phase 0 (workspace setup) | clippy denies pass; `anyhow` absent from core crates |

## Research Flags for Roadmap (phases needing deeper research before execution)

- **GPU phase:** CubeCL ROCm/HIP backend is WIP with raw bindings; wavefront-64 semantics and reduction determinism need a dedicated spike. Confirm CubeCL feature coverage (traits/methods are only partially supported) against the kernels actually required (histogram build, score, leaf estimation).
- **Python packaging phase:** abi3/abi3t status for Python 3.12–3.15 and PyO3/maturin support is in flux (PEP 779/803). Confirm the exact wheel/ABI strategy before committing.
- **Ordered boosting + categorical CTR phase:** the most algorithmically subtle area; warrants a close line-by-line reading of `approx_calcer.cpp` and the online-CTR code with intermediate oracles designed first.

## Sources

- Vendored reference source (HIGH): `catboost-master/catboost/private/libs/quantization/grid_creator.cpp` (border-selection algorithms: GreedyLogSum default + GreedyMinEntropy/MinEntropy/MaxLogSum/Median/Uniform/UniformAndQuantiles); `catboost/private/libs/algo/{approx_calcer.cpp, scoring.cpp, leafwise_scoring.cpp, fold.cpp, learn_context.cpp (RNG seeding `Rand(randomSeed)`), rand_score.cpp, greedy_tensor_search.cpp}`.
- Project docs (HIGH): `.planning/PROJECT.md` (parity bar 1e-5, no-unwrap, thiserror/anyhow, CubeCL backend selection, rocm-only GPU tests, per-backend wheels, Python ≥3.12); `.planning/codebase/CONCERNS.md` (C++ build cost 10–60 min, bindgen pinning, CubeCL/ROCm risk, raw-pointer FFI lifetime issues, GPU inference `#[should_panic]`); `.planning/codebase/TESTING.md` (approx 1e-6 comparisons, oracle fixture model files).
- Float reproducibility literature (HIGH, well-established): IEEE/TOMS "Efficient Reproducible Floating-Point Summation" (Demmel/Ahrens et al.) and "Impacts of floating-point non-associativity on reproducibility for HPC and deep learning" (arXiv:2408.05148) — non-associativity, parallel-reduction-tree-shape dependence, run-to-run/machine-to-machine variability.
- CubeCL / ROCm (MEDIUM, fast-moving): tracel-ai/cubecl + cubecl-hip-sys (ROCm/HIP backend WIP, raw bindgen bindings, partial trait/method support); ROCm wavefront-64 vs CUDA warp-32 correctness trap (CUDA→HIP migration guidance).
- PyO3 / maturin (MEDIUM, evolving): PyO3 free-threading guide (Stable ABI unavailable for free-threaded builds when `Py_GIL_DISABLED` set); PEP 803 (abi3t, 3.15+); maturin issue #3064 (abi3/abi3t wheel support in progress).

---
*Pitfalls research for: Rust rewrite of CatBoost — numerical parity / GPU / bindings*
*Researched: 2026-06-13*

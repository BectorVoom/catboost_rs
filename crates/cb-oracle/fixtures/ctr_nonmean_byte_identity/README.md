# non-mean `.cbm` CTR byte-identity baseline (SPEC-CTRT-16 / E00 / A9)

**CAPTURED_AT_SHA: `f5701caa465916725978947751d7eeb2e59a6a2d`**

Captured on top of the UNCOMMITTED one-hot device wave present in the
working tree at capture time.

Written by `ctr_nonmean_byte_identity_test::capture_ctr_nonmean_baseline`
(`#[ignore]`d; run with `-- --ignored`).

## What is frozen

The `save_cbm` bytes of a HAND-CONSTRUCTED `cb_model::Model` — no
trainer, no RNG, no fixture inputs — carrying:

- two depth-1 oblivious trees, each with exactly one
`ModelSplit::Ctr` (borders `0.25` / `0.5`, prior `0.5/1`,
`shift 0.0`, `scale 1.0`, `target_border_idx 0`), leaf values
`[-0.1, 0.2]` / `[0.05, -0.15]`, leaf weights `[2.0, 3.0]`;
- `bias = 0.125`, `approx_dimension = 1`, no float features, no
class-to-label map;
- a `CtrData` with TWO tables over projection `[0]`, keyed by
`ctr_base_key`:
1. `Borders`, `target_classes_count = 2`,
`int_counts = [[3,7],[11,2],[0,5]]`, `counter_denominator = 0`;
2. `Counter`, `target_classes_count = 0`,
`int_counts = [[10],[13],[5]]`, `counter_denominator = 13`;
both over bucket hashes
`[0x1111111111111111, 0x2222222222222222, 0x3333333333333333]`
(none equal to the `0xFFFFFFFFFFFFFFFF` empty-slot marker).

## Why hand-constructed and not trained

W1-W3 change the trainer's chosen CTR candidate set on purpose
(SPEC-CTRT-11 changes tie-breaks). A trained baseline would drift
with those changes and this gate would silently become vacuous. A
hand-constructed model isolates the SERIALIZER, which is what
SPEC-CTRT-16 gates.

## Do not regenerate

These bytes exist to prove that W4's mean-CTR codec lift
(E19 decode, E20 encode) leaves the NON-MEAN encoding untouched.
Regenerating them after that change turns SPEC-CTRT-16 into a
self-comparison that proves nothing. Still frozen from here on.

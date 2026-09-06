//! Structural guard for the SHARED-MEMORY CUBE INDEPENDENCE rule documented at the
//! top of `kernels.rs`: **every `#[cube]` kernel that declares a `SharedMemory` must
//! end with a top-level `sync_cube()`**.
//!
//! # Why this is a structural test and not a behavioural one
//!
//! The bug this rule prevents is a cross-cube data race that only exists on runtimes
//! which execute cubes SEQUENTIALLY and reuse the one shared buffer across cube
//! iterations — the CubeCL CPU runtime, i.e. this crate's default backend. A
//! behavioural test for it would have to lose a race to fail, so it would pass on a
//! quiet machine and flake on a loaded one; it could never be the thing that keeps
//! the invariant true. The invariant is a property of the kernel SOURCE, so this
//! asserts it directly on the source, deterministically and in microseconds.
//!
//! It is deliberately blunt: it re-derives the kernel list from the file rather than
//! comparing against a hard-coded roster, so a NEW shared-memory kernel is covered
//! the moment it is written instead of the moment someone remembers to list it here.
//!
//! Source/test separation is mandatory (CLAUDE.md): the kernels live in `kernels.rs`,
//! every assertion lives here.

/// The production kernel source, embedded at COMPILE time. `include_str!` resolves
/// relative to this file, so this is immune to the test's working directory — unlike
/// a `std::fs::read` of a `CARGO_MANIFEST_DIR`-relative path, which would silently
/// read the wrong tree in a workspace-root `cargo test` invocation.
const KERNELS_SRC: &str = include_str!("../kernels.rs");

/// One `#[cube]` kernel recovered from the source: its name and its body lines.
struct Kernel {
    name: String,
    /// Body lines between the `fn` signature line and the closing brace, exclusive.
    body: Vec<String>,
}

/// Strip the parts of a line that must not contribute to brace counting: `//`
/// comments and the interiors of string literals. `format!("…{e:?}")` in the host
/// helpers around the kernels would otherwise unbalance the count.
///
/// Handles backslash escapes inside strings. It does NOT model raw strings (`r"…"`)
/// or char literals — `kernels.rs` has neither, and [`brace_counting_is_soundcheck`]
/// fails loudly if that ever stops being true.
fn code_only(line: &str) -> String {
    let mut out = String::new();
    let mut chars = line.chars().peekable();
    let mut in_str = false;
    let mut escaped = false;

    while let Some(c) = chars.next() {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '/' if chars.peek() == Some(&'/') => break,
            _ => out.push(c),
        }
    }

    out
}

/// Recover every `#[cube…]`-attributed function from [`KERNELS_SRC`] by brace
/// matching from the signature line to the top-level closing brace, counting only
/// braces that [`code_only`] leaves standing.
fn cube_kernels() -> Vec<Kernel> {
    let lines: Vec<&str> = KERNELS_SRC.lines().collect();
    let mut out = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if !(trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ")) {
            continue;
        }

        // Walk backwards over the attribute/doc-comment block to find `#[cube…]`.
        let mut j = i;
        let mut is_cube = false;
        while j > 0 {
            let prev = lines[j - 1].trim();
            if !(prev.starts_with('#') || prev.starts_with("//") || prev.is_empty()) {
                break;
            }
            if prev.starts_with("#[cube") {
                is_cube = true;
            }
            j -= 1;
        }
        if !is_cube {
            continue;
        }

        // Brace-match from the signature to the kernel's closing brace.
        let mut depth = 0i32;
        let mut started = false;
        let mut end = None;
        for (k, l) in lines.iter().enumerate().skip(i) {
            for ch in code_only(l).chars() {
                match ch {
                    '{' => {
                        depth += 1;
                        started = true;
                    }
                    '}' => depth -= 1,
                    _ => {}
                }
            }
            if started && depth == 0 {
                end = Some(k);
                break;
            }
        }
        let Some(end) = end else { continue };

        let name = line
            .split("fn ")
            .nth(1)
            .and_then(|rest| rest.split(['<', '(']).next())
            .unwrap()
            .trim()
            .to_string();

        out.push(Kernel {
            name,
            body: lines[i + 1..end].iter().map(|l| l.to_string()).collect(),
        });
    }

    out
}

/// Self-check for the untokenized brace matching in [`cube_kernels`]: over the whole
/// file, the braces [`code_only`] leaves standing must balance to exactly zero and
/// must never dip below it.
///
/// This is what keeps the real guard honest. If [`code_only`] ever mishandles the
/// source — a raw string (`r"…{…"`), a char literal `'{'`, a multi-line string — the
/// count drifts, [`cube_kernels`] mis-slices kernel bodies, and the guard below could
/// silently stop seeing the kernels it is supposed to police. Failing here says
/// "teach the parser about this construct" instead of quietly going vacuous.
#[test]
fn brace_counting_is_soundcheck() {
    let mut depth = 0i32;
    for (n, line) in KERNELS_SRC.lines().enumerate() {
        for ch in code_only(line).chars() {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        assert!(
            depth >= 0,
            "brace depth went negative at kernels.rs:{} — `code_only` is mis-reading \
             the source and `cube_kernels` cannot be trusted",
            n + 1
        );
    }
    assert_eq!(
        depth, 0,
        "braces in kernels.rs do not balance under `code_only` (ended at depth \
         {depth}) — teach that sanitizer about the construct it is missing before \
         relying on `cube_kernels`"
    );
}

/// The guard itself. Every `#[cube]` kernel that declares a `SharedMemory` must end
/// with a `sync_cube()` at the kernel's TOP LEVEL (4-space indent — i.e. not nested
/// inside an `if`/`while`, where only some units would reach it and the barrier would
/// deadlock instead of synchronizing).
///
/// A plain `#[cube]` HELPER (as opposed to a `#[cube(launch)]` kernel entry point) may
/// return a value instead of `()`, in which case its literal last line is the bare
/// tail-expression variable, not `sync_cube();` — `plane_carry_scan` is the first such
/// case. That shape is still accepted, but only in the exact form that preserves the
/// same guarantee: the barrier must be the line immediately BEFORE that bare
/// identifier, i.e. still the last thing touching `SharedMemory` before control
/// returns to the caller. This is a structural generalization of the same rule, not a
/// named exception — it applies to any future value-returning shared-memory helper
/// shaped the same way, and nothing that pattern-matches a specific function name.
///
/// See SHARED-MEMORY CUBE INDEPENDENCE at the top of `kernels.rs` for why: CubeCL
/// shares one `SharedMemory` allocation BETWEEN cubes, and the CPU runtime reuses it
/// across sequential cube iterations, so without this barrier a unit that finishes
/// early overwrites slots a slower unit is still reading for the previous cube.
#[test]
fn every_shared_memory_kernel_ends_with_a_trailing_sync_cube() {
    let kernels = cube_kernels();

    // Guard the guard: if the parser stops recognizing kernels (a syntax change, a
    // rename), an empty roster would make this test vacuously green.
    assert!(
        kernels.len() >= 20,
        "recovered only {} #[cube] kernels from kernels.rs — the parser in \
         `cube_kernels` has drifted and this guard would pass vacuously",
        kernels.len()
    );

    let shared: Vec<&Kernel> = kernels
        .iter()
        .filter(|k| k.body.iter().any(|l| l.contains("SharedMemory")))
        .collect();

    assert!(
        !shared.is_empty(),
        "no shared-memory kernels found in kernels.rs — the filter in this guard has \
         drifted and it would pass vacuously"
    );

    // A bare tail-expression return: a lone identifier, no parens/operators/semicolon
    // (e.g. `carry`). Anything else on this line is not the value-returning-helper
    // shape and must fall through to the ordinary kernel check.
    fn is_bare_tail_expr(line: &str) -> bool {
        let t = line.trim();
        !t.is_empty()
            && t.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
            && t.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
    }

    let offenders: Vec<&str> = shared
        .iter()
        .filter(|k| {
            let mut tail = k
                .body
                .iter()
                .rev()
                .filter(|l| !l.trim().is_empty() && !l.trim().starts_with("//"));
            let last = tail.next();
            // Exactly 4 spaces of indent == the kernel's top level.
            if last.map(|l| l.as_str()) == Some("    sync_cube();") {
                return false;
            }
            // Value-returning helper shape: bare tail expression, with the barrier as
            // the line immediately before it.
            if last.is_some_and(|l| is_bare_tail_expr(l)) {
                let prev = tail.next();
                if prev.map(|l| l.as_str()) == Some("    sync_cube();") {
                    return false;
                }
            }
            true
        })
        .map(|k| k.name.as_str())
        .collect();

    assert!(
        offenders.is_empty(),
        "these shared-memory kernels do not end with a top-level `sync_cube()`, so \
         they race across cube iterations on the sequential CPU runtime: {offenders:?}\n\
         Add `sync_cube();` as the last statement of each — see SHARED-MEMORY CUBE \
         INDEPENDENCE at the top of kernels.rs."
    );

    println!(
        "[cube-independence] {} of {} #[cube] kernels use SharedMemory; all end with \
         a top-level sync_cube()",
        shared.len(),
        kernels.len()
    );
}




//! Profiling harness for the CPU-runtime derivative kernels — the measurement side of
//! `bench/perf_param_cpu/KERNEL-DESIGN-PASS.md` §3-§5.
//!
//! ```text
//! cargo run -p cb-backend --release --example der_kernel_profile -- e2e 15
//! cargo run -p cb-backend --release --example der_kernel_profile -- vec 11
//! ```
//!
//! `e2e` times `CpuBackend::compute_gradients` per loss (what the trainer pays);
//! `vec` times the kernels alone with `client.profile` across unit width × vector
//! width, checking every configuration is bit-identical to the scalar one-unit launch.
//! Repetitions are interleaved across configurations (the schedule discipline of
//! `bench/perf_param_cpu/FINDINGS.md`) and the JIT is warmed before timing.
//!
//! For a host-side attribution (which copies, which waits), sample the `e2e` mode with
//! the platform profiler — on macOS `sample <pid>`, on Linux `perf record` — as the
//! design pass did; the CubeCL `profile-tracy` feature is the in-process alternative.

#[cfg(feature = "cpu")]
mod harness {
    use std::error::Error;
    use std::time::{Duration, Instant};

    use cubecl::client::ComputeClient;
    use cubecl::cpu::{CpuDevice, CpuRuntime};
    use cubecl::prelude::*;

    use cb_backend::kernels::{
        gradient_kernel, huber_gradient_kernel, logloss_gradient_kernel, quantile_gradient_kernel,
    };
    use cb_compute::{Loss, Runtime as _};

    type Client = ComputeClient<CpuRuntime>;
    type Res<T> = Result<T, Box<dyn Error>>;

    fn ms(d: Duration) -> f64 {
        d.as_secs_f64() * 1e3
    }

    fn median(samples: &[Duration]) -> Duration {
        let mut sorted = samples.to_vec();
        sorted.sort();
        sorted.get(sorted.len() / 2).copied().unwrap_or_default()
    }

    /// Deterministic sign-mixed inputs: logits in `[-2, 2)` and `{0, 1}` labels.
    fn data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let approx: Vec<f64> = (0..n).map(|_| next() * 4.0 - 2.0).collect();
        let target: Vec<f64> = (0..n).map(|_| if next() > 0.5 { 1.0 } else { 0.0 }).collect();
        (approx, target)
    }

    fn upload(client: &Client, values: &[f64]) -> cubecl::server::Handle {
        client.create(cubecl::bytes::Bytes::from_elems(values.to_vec()))
    }

    fn geometry(lanes: usize, units: u32) -> (CubeCount, CubeDim) {
        let cubes = (lanes as u32).div_ceil(units).max(1);
        (CubeCount::Static(cubes, 1, 1), CubeDim::new_1d(units))
    }

    pub fn e2e(reps: usize) -> Res<()> {
        let backend = cb_backend::CpuBackend;
        let losses: [(&str, Loss); 5] = [
            ("Rmse", Loss::Rmse),
            ("Logloss", Loss::Logloss),
            ("LogCosh", Loss::LogCosh),
            ("Huber", Loss::Huber { delta: 1.0 }),
            ("Quantile", Loss::Quantile { alpha: 0.5, delta: 1e-6 }),
        ];
        // 1_000_003 is deliberately not a multiple of any vector width.
        for n in [100_000usize, 1_000_000, 1_000_003] {
            let (approx, target) = data(n);
            for (_, loss) in &losses {
                backend.compute_gradients(loss, &approx, &target, 1)?;
            }
            let mut times: Vec<Vec<Duration>> = vec![Vec::new(); losses.len()];
            for _ in 0..reps {
                for ((_, loss), samples) in losses.iter().zip(times.iter_mut()) {
                    let start = Instant::now();
                    let ders = backend.compute_gradients(loss, &approx, &target, 1)?;
                    samples.push(start.elapsed());
                    std::hint::black_box(ders);
                }
            }
            println!("\n== compute_gradients n={n} (median of {reps}) ms");
            for ((name, _), samples) in losses.iter().zip(&times) {
                let min = samples.iter().min().copied().unwrap_or_default();
                println!("  {name:<9} {:.3}  (min {:.3})", ms(median(samples)), ms(min));
            }
        }
        Ok(())
    }

    pub fn vec_sweep(reps: usize) -> Res<()> {
        let client: Client = <CpuRuntime as cubecl::Runtime>::client(&CpuDevice);
        let kernels = ["rmse", "huber", "quantile", "logloss"];
        let widths = [1u32, 2, 3, 4, 8];
        let vectors = [1usize, 8];
        for n in [100_000usize, 300_000, 1_000_000, 3_000_000] {
            let (approx, target) = data(n);
            let approx_h = upload(&client, &approx);
            let target_h = upload(&client, &target);
            let alpha_h = upload(&client, &[0.7]);
            let delta_h = upload(&client, &[0.05]);
            cubecl::future::block_on(client.sync())?;

            let run = |kernel: usize, units: u32, vector: usize| -> Res<(Vec<u8>, Duration, Duration)> {
                let out = client.empty(n * std::mem::size_of::<f64>());
                let (count, dim) = geometry(n / vector, units);
                let start = Instant::now();
                let launch = || {
                    let a = unsafe { ArrayArg::from_raw_parts(approx_h.clone(), n) };
                    let t = unsafe { ArrayArg::from_raw_parts(target_h.clone(), n) };
                    let o = unsafe { ArrayArg::from_raw_parts(out.clone(), n) };
                    let p1 = unsafe { ArrayArg::from_raw_parts(alpha_h.clone(), 1) };
                    let p2 = unsafe { ArrayArg::from_raw_parts(delta_h.clone(), 1) };
                    match kernel {
                        0 => gradient_kernel::launch::<f64, CpuRuntime>(&client, count.clone(), dim, vector, a, t, o),
                        1 => huber_gradient_kernel::launch::<f64, CpuRuntime>(&client, count.clone(), dim, vector, a, t, o, p1),
                        2 => quantile_gradient_kernel::launch::<f64, CpuRuntime>(&client, count.clone(), dim, vector, a, t, o, p1, p2),
                        _ => logloss_gradient_kernel::launch::<f64, CpuRuntime>(&client, count.clone(), dim, vector, a, t, o),
                    }
                };
                let (_, profile) = client.profile(launch, "der").map_err(|e| format!("{e:?}"))?;
                let kernel_time = cubecl::future::block_on(profile.resolve()).duration();
                let bytes = client.read_one(out)?;
                Ok((bytes.to_vec(), kernel_time, start.elapsed()))
            };

            let mut configs = Vec::new();
            for kernel in 0..kernels.len() {
                for &units in &widths {
                    for &vector in &vectors {
                        configs.push((kernel, units, vector));
                    }
                }
            }
            // Warm the JIT and pin bit-identity against the scalar single-unit launch.
            let mut reference = Vec::new();
            for kernel in 0..kernels.len() {
                reference.push(run(kernel, 1, 1)?.0);
            }
            for &(kernel, units, vector) in &configs {
                let (bytes, _, _) = run(kernel, units, vector)?;
                if reference.get(kernel) != Some(&bytes) {
                    return Err(format!(
                        "{} at w={units} vec={vector} is not bit-identical to the scalar launch",
                        kernels.get(kernel).copied().unwrap_or("?")
                    )
                    .into());
                }
            }
            let mut kernel_times: Vec<Vec<Duration>> = vec![Vec::new(); configs.len()];
            let mut total_times: Vec<Vec<Duration>> = vec![Vec::new(); configs.len()];
            for _ in 0..reps {
                for (config, (kt, tt)) in configs.iter().zip(kernel_times.iter_mut().zip(total_times.iter_mut())) {
                    let (_, kernel_time, total) = run(config.0, config.1, config.2)?;
                    kt.push(kernel_time);
                    tt.push(total);
                }
            }
            println!("\n== n={n} (median of {reps}; all configurations bit-identical to scalar w=1)");
            println!("{:<9} {:>3} {:>3} | {:>9} {:>9}", "kernel", "w", "vec", "kern ms", "total ms");
            for (config, (kt, tt)) in configs.iter().zip(kernel_times.iter().zip(&total_times)) {
                println!(
                    "{:<9} {:>3} {:>3} | {:>9.3} {:>9.3}",
                    kernels.get(config.0).copied().unwrap_or("?"),
                    config.1,
                    config.2,
                    ms(median(kt)),
                    ms(median(tt))
                );
            }
        }
        Ok(())
    }
}

#[cfg(feature = "cpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("e2e");
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    match mode {
        "e2e" => harness::e2e(reps),
        "vec" => harness::vec_sweep(reps),
        other => Err(format!("unknown mode `{other}`; expected e2e | vec").into()),
    }
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("der_kernel_profile measures the CPU runtime; build with the default `cpu` feature");
}

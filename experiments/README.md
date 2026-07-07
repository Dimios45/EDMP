# Experiments

Benchmark launchers for the paper *"The Prior Is Not the Bottleneck."* Each script
is self-locating (`cd`s to the repo root via its own path), shards across 8 workers
with `OMP_NUM_THREADS=1`, writes per-worker logs to `logs/`, and merges results to
`results/<tag>/summary.json`. Run from anywhere, e.g.:

```bash
bash experiments/feasibility_repair.sh
```

Analysis / rendering entry points stay at the repo root because they import
`run_worker` as a module: `diag_failures.py`, `analyze_smoothness.py`,
`render_scenes.py`, `train_gpd.py`, `train_gpd_cond.py`.

## Script → paper map

| Script | Paper § | What it measures | Results |
|---|---|---|---|
| `baselines.sh` | Tab. §5.6 | GPD / GPD+repair / EDMP × global/hybrid/both | `results/baseline_*` |
| `reproduction_guidance_tuning.sh` | §Reproduction | K × guidance-scale sweep closing the gap to the paper | `results/reproduction_*` |
| **Prior-side (§3 — none help)** | | | |
| `prior_capacity.sh` | §3.2 | n=8 vs n=16 Bernstein capacity | `results/capacity_*` |
| `prior_smoothness.sh` | §3.2 | smoothness-penalty sweep on n=16 | `results/smoothness_lam*` |
| `prior_conditioning.sh` | §3.3 | scene-conditioned denoiser + CFG | `results/conditioning_cfgw*` |
| `prior_conditioning_ablation.sh` | §3.3 | CFG-only / prior-only ablation | `results/conditioning_{cfg_only,prior_only}` |
| `prior_long_training.sh` | §3.5 | 1M-step (5×) extended training | `results/extended_*` |
| **Feasibility machinery (§4 — the levers)** | | | |
| `feasibility_repair.sh` | §4.3 | trajopt repair: off / densify-only / full | `results/repair_{off,densify,full}` |
| `feasibility_repair_pareto.sh` | §4.6 | trajopt hyperparameter Pareto (iters/densify/mid-w) | `results/repair_pareto_*` |
| `feasibility_sphere_collision.sh` | §4.7 | exact sphere-SDF vs AABB-proxy objective | `results/repair_sphere_*` |
| `feasibility_naive_seed.sh` | §4.4 | linear-seed control (is the prior needed?) | `results/linearseed_*` |
| `feasibility_cross_planner.sh` | §4.5, §4.1 | EDMP + same repair; failure-mode shift | `results/edmp_*`, `results/failure_modes_repair.json` |
| **Statistics (§4.3, §3.6)** | | | |
| `stats_seeds_base_repair.sh` | §4.3 | 5-seed CIs: baseline vs repair | `results/seed_{baseline,repair}_s*` |
| `stats_seeds_prior.sh` | §3.3/§3.5 | per-intervention seeds: conditioning, extended training | `results/seed_{conditioning,extended}_s*` |

## Analysis entry points (repo root)

| Script | Paper § | Output |
|---|---|---|
| `diag_failures.py [N] [base\|repair]` | §4.1 | `results/failure_modes{,_repair}.json` |
| `analyze_smoothness.py [per_type]` | §3.2 | `results/smoothness_stats.json`, `smoothness_jerk.png` |
| `render_scenes.py` | §3 (Fig. 2) | `results/env_montage.png` |

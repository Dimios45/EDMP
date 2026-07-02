# GPD Experiments — Full Work Log

Complete record of the GPD (Guided Polynomial Diffusion) reproduction + improvement work inside the EDMP codebase. Companion to [RESEARCH_PLAN.md](RESEARCH_PLAN.md) (forward-looking plan) and [BLOG.md](BLOG.md) (original implementation write-up).

**Hardware:** RTX 4090 (24 GB), 32-core CPU. **Robot:** Franka Panda (7-DoF). **Paper:** [arXiv:2501.18229](https://arxiv.org/html/2501.18229v1).

---

## 1. Environment setup

- **Env:** `uv venv .venv` (Python 3.10). NOTE: `uv pip install torch` **deadlocks** extracting the CUDA wheels on this box — use the venv's `pip` instead (`python -m ensurepip --upgrade` then `python -m pip install ...`). uv is fine for small packages.
- **Installed:** torch 2.12.1+cu130 (CUDA OK on 4090), torchvision, h5py, einops, scipy, matplotlib, pybullet, wandb, ruamel.yaml<0.18, tqdm, gdown, autolab_core, geometrout 0.0.3.4, urchin, trimesh.
- **robofin** @ branch `v0.0.1`, installed editable (`pip install -e robofin --no-deps`).
- **ikfast** (`ikfast-pybind`, provides `import ikfast_franka_panda`): built from source with `CMAKE_POLICY_VERSION_MINIMUM=3.5` set in the env (works around a CMake-4 policy failure).

### Data & models
| Asset | Location |
|---|---|
| MPInets training data (15.8 GB, `global_solutions`+`hybrid_solutions`, 3.27M each) | `/mnt/hdd/mpinets_hybrid_training_data/train/train.hdf5` → symlinked `data/mpinets_dataset.hdf5` |
| EDMP benchmark scenes | `datasets/{both,global,hybrid}_solvable_problems.pkl` (Drive) |
| EDMP pretrained model | `models/TemporalUNetModel255_N50/` (Drive) |
| GPD training data (preprocessed) | `gpd_train_combined.hdf5` → `control_points (6540000, 7, 8)` |

Preprocess (run scripts under `gpd/` **as modules** from repo root, else `from diffusion.diffusion import` breaks):
```bash
python -m gpd.preprocess_data --input data/mpinets_dataset.hdf5 --key hybrid_solutions --output gpd_train_hybrid.hdf5 --transpose --n_control 8
python -m gpd.preprocess_data --input data/mpinets_dataset.hdf5 --key global_solutions --output gpd_train_global.hdf5 --transpose --n_control 8
python -m gpd.merge_datasets --inputs gpd_train_hybrid.hdf5 gpd_train_global.hdf5 --output gpd_train_combined.hdf5
```

---

## 2. Training

`train_gpd.py` trains a TemporalUNet (dims (32,64,128,256), input_dim=7, time_dim=32, ~3.95M params) in Bernstein control-point space. NOTE: one "epoch" = one 2048-sample mini-batch step (not a full pass); 20k "epochs" ≈ 6 real passes over 6.54M.

| Model | Steps | Schedule | Final loss | Location |
|---|---|---|---|---|
| GPDModel64_N8 (baseline) | 20k → 200k | linear thr 0.02 | 0.527 → 0.481 | `models/GPDModel64_N8/` |
| 20k snapshot | 20k | linear thr 0.02 | 0.527 | `models_20k/GPDModel64_N8/`, `models/GPDModel64_N8_ckpt20k/` |
| schedule-recalibration model | 20k | linear thr **0.08** | 0.433 | `models_d1/GPDModel64_N8/` |

```bash
# baseline (resumes from existing losses.npy length)
python train_gpd.py --dataset gpd_train_combined.hdf5 --T 64 --n_control 8 --epochs 200000 --batch_size 2048
# schedule-recalibration (calibrated schedule, separate dir)
python train_gpd.py --dataset gpd_train_combined.hdf5 --T 64 --n_control 8 --epochs 20000 --variance_thresh 0.08 --model_dir ./models_d1/ --no_wandb
```
Speed: ~14–28 steps/s on the 4090; 20k ≈ 16 min, 200k ≈ 2.5 hr. wandb project `GPD_denoiser`.

---

## 3. Benchmark methodology

`run_worker.py` (headless PyBullet) evaluates GPD/EDMP on the 4 MPInets scene types. Success = `env.benchmark_trajectory()` (collision-free + reaches goal). Added flags this session:

| Flag | Meaning |
|---|---|
| `--per_type N` | balanced N scenes/type |
| `--caps a,b,c,d` | explicit per-type caps (natural dist = `600,300,300,600`) |
| `--guides "1"` | guide ids (single-guide GPD = single guide) |
| `--batch_per K` | candidates per guide (K = n_guides × batch_per) |
| `--no_stitch` | pick best single candidate (paper GPD-NG) |
| `--linear_stitch` | linear bridges instead of RRT-Connect when stitching |
| `--model_dir DIR` | parent dir containing `GPDModel{T}_N{n}/` |
| `--variance_thresh v` | inference noise schedule (must match training) |

**Important — two different aggregation axes (not comparable):**
- **Paper** reports per *solvability* set (global/hybrid/both solvable, 1800 each), aggregate.
- **BLOG / our per-type tables** report per *environment* type (tabletop/cubby/merged_cubby/dresser).
- To compare to the paper, aggregate over the full set with **natural weights** (tabletop 600, cubby 300, merged_cubby 300, dresser 600).

---

## 4. Results

### 4a. Training-duration study (12-guide + linear-stitch config, 100/type = 400 scenes)
| Scene | 20k | 200k | BLOG |
|---|---|---|---|
| tabletop | 94.0% | 98.0% | 96.2% |
| cubby | 31.0% | 32.0% | 40.7% |
| merged_cubby | 17.0% | 24.0% | 34.7% |
| dresser | 34.0% | 33.0% | 43.3% |
| **Overall (balanced)** | **44.0%** | **46.8%** | 59.1% |

→ **10× training = +2.8 pp**, concentrated in easy/prior-limited scenes; constrained-scene gap intact. **Training duration is not the bottleneck.**

### 4b. Reproduction sprint — single-guide GPD (1 guide, K=32, best-of-K, no stitch)
| Run | nat-wtd SR | per-type (tab/cub/merg/dres) |
|---|---|---|
| Exp 1 (200k, 300 scenes) | **62.3%** | 96/56/44/41 |
| 2a (20k, 150-scene subset) | 66.0%* | 100/72/28/48 |
| 2b schedule-recalibration (20k, thr 0.08) | 53.3% | 84/44/16/46 |
| 12-guide + linear stitch (200k) | ≈53% | — |
| **Paper single-guide GPD** | **72.9%** | — |

\*smaller/easier subset than Exp 1.

→ **single-guide GPD reproduced to ~10 pp.** The earlier ~38 pp "gap" was mostly a **config artifact** (12-guide + linear stitch underperforms single-guide best-of-K). **schedule-recalibration schedule fix refuted** (66.0→53.3, worse on all types despite lower loss; the low-noise ᾱ_T≈0.52 regime is an implicit smoothness prior — ε-loss misleads).

### 4c. collision-aware stitching — stitching ablation (single-guide GPD, 200k, 150-scene natural subset, 24 sharded workers)
| Bridge mode | nat-wtd SR | tab | cub | merg | dres |
|---|---|---|---|---|---|
| no-stitch (best-of-K) | **62.0%** | 100 | 60 | 28 | 42 |
| linear-stitch | 54.0% | 96 | 40 | 8 | 42 |
| **RRT-Connect-stitch** | **62.0%** | 100 | 72 | 24 | 38 |

Δ: RRT vs linear **+8.0 pp**; no-stitch vs linear **+8.0 pp**; RRT vs no-stitch **+0.0 pp**.

**Findings:**
- **Linear stitching is confirmed harmful** (54% vs 62% no-stitch) — third independent confirmation.
- **RRT-Connect repairs the damage** (+8 pp over linear, back to no-stitch level) and **improves cubby (+12 pp: 60→72)** where bridging genuinely helps.
- **But RRT only *ties* best-of-K overall (62%)** — it does NOT yet reproduce the paper's GPD-stitched jump (72.9→92.8). It slightly underperforms on merged_cubby/dresser (within the ±~18 pp CI of n=25/type).
- **Why no GPD-stitched-level gain (hypotheses → next work):** (i) RRT uses the AABB intersection-volume *proxy* as its collision checker, not true mesh/PyBullet collision — bridges valid under the proxy can still collide in reality; (ii) single-guide K=32 may lack the candidate *diversity* GPD-stitched relies on; (iii) stitch-target heuristic (dist_threshold, earliest-forward) is crude. Next: a faithful collision checker inside RRT + higher candidate diversity.

Note: at 24 sharded workers the GPU (denoising) becomes the bottleneck (99% util) so per-scene plan-time inflates (~10 s) but wall-clock throughput is ~2–3× a single worker; an intermediate worker count (~8–12) is likely the sweet spot.

---

### 4d. guide-diversity — diversity × stitching (GPD, 200k, 150-scene natural)
| config | nat-wtd | tab | cub | merg | dres |
|---|---|---|---|---|---|
| 1-guide K=32 no-stitch | 62.0% | 100 | 60 | 28 | 42 |
| 1-guide K=32 RRT | 62.0% | 100 | 72 | 24 | 38 |
| 12-guide K=120 no-stitch | 61.3% | 98 | 60 | 28 | 42 |
| 12-guide K=120 RRT | 60.0% | 100 | 64 | 24 | 38 |

**Conclusion: diversity does NOT unlock stitching.** All four configs plateau at ~60–62%. More guides/candidates (K=32→120) didn't help with or without RRT. → the bottleneck is not candidate diversity nor the stitch *mechanism*; the remaining suspect is the **collision checker** (RRT certifies bridges with the loose AABB intersection-volume proxy, not true mesh/PyBullet collision).

### Major efficiency fix (answers "why CPU-only / idle GPU / slow")
`lib/environment.py:benchmark_trajectory` had a **`time.sleep(0.4)` per waypoint** — pure wasted wall-clock in headless mode (~20 s/scene). Gated behind `self.gui`. Result: **~0.6 s/scene** (was ~10–20 s), a ~20× speedup. Combined with sharding (`--num_workers` + `OMP_NUM_THREADS=1`), benchmarks are now fast and GPU-bound.

### 4e. ⭐ collision-aware stitching-faithful — RRT + PyBullet collision checker (GPD, 200k, 150-scene natural)
| config | nat-wtd | tab | cub | merg | dres |
|---|---|---|---|---|---|
| 1g K=32 RRT + AABB | 62.0% | 100 | 72 | 24 | 38 |
| **1g K=32 RRT + PyBullet** | **76.0%** | 98 | 88 | 40 | 66 |
| 12g K=120 RRT + AABB | 60.0% | 100 | 64 | 24 | 36 |
| **12g K=120 RRT + PyBullet** | **76.0%** | 100 | 76 | 52 | 64 |
| 1g no-stitch (ref) | 62.0% | 100 | 60 | 28 | 42 |

**⭐ The faithful collision checker breaks the 62% plateau → 76% (+14 pp).** The AABB intersection-volume proxy was the limiter — it certified bridges that actually collide in PyBullet. With true `getContactPoints` checking in RRT, gains land on the previously-weak constrained scenes: **dresser 38→66, merged_cubby 24→52, cubby 72→88.** Now **above the paper's single-guide GPD (72.9%)**. Implemented via `RobotEnvironment.configs_free()` + `stitch(collision_fn=...)`; enable with `run_worker.py --pybullet_collision`. (n=25/type for cubby/merged → ±~18 pp CI; dresser n=50; gains large enough to be real. PyBullet check stays fast: ~0.1–0.3 s/bridge.)

### 4f. Full paper replication (3 configs × 3 datasets, 600 scenes/dataset, paper metric)
See **REPLICATION.md** for the step-by-step log. Final 3×3 (ours / paper):
| | global | hybrid | both |
|---|---|---|---|
| single-guide GPD | 50.8 / 65.5 | 57.8 / 72.9 | 54.2 / 73.0 |
| 7-guide GPD | 56.4 / 81.9 | 63.8 / 90.9 | 64.0 / 90.2 |
| GPD-stitched | 63.5 / 87.0 | 70.7 / 92.8 | 71.7 / 92.6 |

**Avg −21 pp.** Method + ordering (GPD-stitched>7G>1G) replicate; absolute SR does not. Gap is ~half tunable (K=32→128 +6 pp; 2× guidance +4.5 pp) and ~half model-weights/guide-tuning. Faithful-collision RRT stitching is the proven +13–14 pp win; candidate-pool diversity / longer training / noise-schedule recalibration all tested and don't help.

## 5. Key findings
1. **Training duration is not the bottleneck** (+2.8 pp for 10×).
2. **single-guide GPD reproduces to ~10 pp** with the correct single-guide best-of-K config.
3. **schedule-recalibration (terminal-SNR recalibration) refuted** — low-noise schedule is an implicit smoothness prior; ε-loss ≠ planning success.
4. **Linear-bridge stitching is harmful** (can make trajectories worse than the best raw candidate). The paper's headline gain comes from RRT-Connect stitching → implemented (§6).

---

## 6. Code changes this session
- `run_worker.py`: added `--per_type --caps --guides --batch_per --no_stitch --linear_stitch --model_dir --variance_thresh`; stitch call now passes `use_rrt`.
- `gpd/dataset.py`, `train_gpd.py`: `--variance_thresh` / `--schedule` (linear|cosine) parametrized noise schedule.
- `gpd/stitch.py`: **added self-contained RRT-Connect** local planner (`rrt_connect`, `_configs_free`, `_edge_free`) using the guide's intersection-volume cost as the collision checker; `stitch(..., use_rrt=True)` bridges with RRT-Connect (linear fallback on failure).

## 7. Artifact map
- Models: `models/GPDModel64_N8/` (200k), `models_20k/`, `models_d1/`.
- Results JSON: `results/gpd_mini.json`, `gpd_100_20k.json`, `gpd_100_200k.json`, `repro_gpd1g_hybrid.json`, `abl_gpd1g_base20k.json`, `abl_gpd1g_d1.json`, `d2_{nostitch,linear,rrt}.json`.
- Logs: `_dl_logs/*.log`.

## 8. Reproduce key runs
```bash
source .venv/bin/activate
# single-guide GPD reproduction vs paper (natural dist)
python run_worker.py --method gpd --guides 1 --batch_per 32 --no_stitch --caps 100,50,50,100 --out results/repro_gpd1g_hybrid.json
# collision-aware stitching stitching ablation
python run_worker.py --method gpd --guides 1 --batch_per 32 --no_stitch     --caps 50,25,25,50 --out results/d2_nostitch.json
python run_worker.py --method gpd --guides 1 --batch_per 32 --linear_stitch --caps 50,25,25,50 --out results/d2_linear.json
python run_worker.py --method gpd --guides 1 --batch_per 32                  --caps 50,25,25,50 --out results/d2_rrt.json
```
Aggregate natural-weighted: weight per-type SR by {tabletop:600, cubby:300, merged_cubby:300, dresser:600}.

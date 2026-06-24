# Improving Guided Polynomial Diffusion (GPD): Diagnosis, Research Directions, and Paper Plan

**Base:** GPD ([arXiv:2501.18229](https://arxiv.org/html/2501.18229v1)) implemented inside the EDMP codebase.
**Hardware:** RTX 4090. **Robot:** Franka Panda (7-DoF). **Benchmark:** MPiNets scenes (tabletop / cubby / merged_cubby / dresser).

---

## 1. Reproduction status (our results)

We trained `GPDModel64_N8` (TemporalUNet, dims (32,64,128,256), ~3.95M params, T=64, 8 Bernstein control points) on the 6.54M-trajectory MPiNets set (global + hybrid solutions projected to Bernstein control points), then benchmarked at **100 scenes/type (400 scenes)** with 12 guides × 10 samples + linear-bridge stitching.

| Scene | 20k-step model | 200k-step model | Δ (10× train) | BLOG GPD | Paper GPDS |
|---|---|---|---|---|---|
| tabletop | 94.0% | **98.0%** | +4.0 | 96.2% | — |
| cubby | 31.0% | **32.0%** | +1.0 | 40.7% | — |
| merged_cubby | 17.0% | **24.0%** | +7.0 | 34.7% | — |
| dresser | 34.0% | **33.0%** | −1.0 | 43.3% | — |
| **Overall** | **44.0%** | **46.8%** | **+2.8** | 59.1% | 87–92% (solvable-type slicing) |

Plan time (200k, uncontended): **1.55 s/scene avg** — faster than BLOG's 2.02 s.

**Key observation (control experiment result):** training 10× longer (20k→200k steps; loss 0.527→0.481, ≈9% relative) yields only **+2.8 pp overall**, concentrated in the prior-quality-limited scenes (tabletop +4, merged_cubby +7) while **cubby and dresser stay flat**. The constrained-scene gap to BLOG *remains large* (cubby −8.7, merged_cubby −10.7, dresser −10.3). **This confirms training duration is NOT the bottleneck — the constrained-scene deficit is structural** (noise-schedule calibration / global smoothness / linear-bridge stitching), validating the directions below.

---

## 1b. Reproduction sprint (added after experiments)

We targeted **one** paper baseline — **GPD-1G** (single guide, K=32 candidates, best-of-K selection, **no stitching**) — to validate the core method, then tested the D1 schedule idea as a controlled ablation. All on hybrid-solvable, natural-distribution, reported as natural-weighted SR.

| Run | Config | nat-wtd SR | per-type (tab/cub/merg/dres) |
|---|---|---|---|
| **Exp 1** (300 scenes) | GPD-1G, 200k, thr 0.02 | **62.3%** | 96 / 56 / 44 / 41 |
| 2a (150 scenes) | GPD-1G, 20k, thr 0.02 | 66.0%* | 100 / 72 / 28 / 48 |
| 2b (150 scenes) | GPD-1G, 20k, **thr 0.08 (D1)** | 53.3% | 84 / 44 / 16 / 46 |
| (prior) | **12-guide + linear stitch**, 200k | ≈53% | 98 / 32 / 24 / 33 |
| Paper | GPD-1G | **72.9%** | — |

*2a used a smaller/easier 150-scene subset than Exp 1's 300; not directly comparable to Exp 1, but 2a vs 2b *are* same-scene and so a clean D1 ablation.

**Findings:**
1. **GPD-1G reproduced to within ~10 pp** (Exp 1 62.3% vs paper 72.9%). The previously-alarming ~38 pp "gap" was largely a **configuration artifact**: our headline runs used 12 guides + linear-bridge stitching (≈53%), which *underperforms* the paper's simple single-guide best-of-K. Residual ~10 pp likely = guide-hyperparameter / K / success-criterion / guide-choice differences.
2. **D1 (schedule recalibration) is REFUTED.** Same scenes, matched 20k budget: thr 0.02 → 0.08 *lowered* SR on all 4 types (66.0 → 53.3 nat-wtd) despite *lower* training loss (0.53 → 0.43). Mechanism: ᾱ_T≈0.52 keeps the reverse process near-identity → lightly polishes smooth, demonstration-like trajectories. Raising terminal noise injects more reverse-process variance → less-smooth, more-colliding paths. **The "miscalibration" is an implicit smoothness prior; ε-loss is a poor proxy for planning success.**
3. **Linear-bridge stitching is harmful**, not just suboptimal: single-guide best-of-K (no stitch) **beats** 12-guide + linear stitch. → confirms D2 from a second angle.

**Reprioritization:** D1 ↓ (drop the naive schedule fix; the low-noise regime is good). **D2 ↑↑** (fix or remove stitching) is now the top lever. D3 (scene-conditioning) and D4 (representation) unchanged.

---

## 2. Diagnosis — why GPD underperforms on constrained scenes

Four root causes, ordered by expected impact / effort ratio.

### 2.1 🔴 Noise schedule is mis-calibrated for the short T=64 chain (headline finding)
The variance schedule is linear `β = linspace(0, 0.02, T+1)`, inherited from EDMP's T=255 **without rescaling** for the 4× shorter chain. Consequence (verified numerically):

| Config | ᾱ_T (signal retained at final forward step) |
|---|---|
| EDMP T=255, thresh 0.02 | 0.076 → ≈ pure noise ✓ |
| **GPD T=64, thresh 0.02** | **0.520 → still 72% signal ✗** |
| GPD T=64, thresh 0.08 | 0.069 → ≈ pure noise ✓ |

At t=T the model is trained on inputs that are still 72% clean signal, but at **inference it must denoise from pure `N(0,I)`** (`denoise_guided_poly` initialises `randn`). This is an out-of-distribution train/inference gap that caps sample quality and explains the high loss floor (~0.48 MSE on ε-prediction). **Fix is ~1 line** (rescale thresh to ≈0.08, or switch to a cosine/SNR-matched schedule).

### 2.2 🟠 Stitching uses linear bridges, not the paper's RRT-Connect
`gpd/stitch.py` bridges trajectory segments with 3-point **linear interpolation**, which can pass straight through obstacles — so a "stitched" trajectory can still collide. The paper's Algorithm 2 uses **RRT-Connect** as the local planner (guaranteed collision-free bridges). This directly costs success on cubby/merged_cubby where stitching is most needed.

### 2.3 🟠 The denoiser is environment-unconditioned
TemporalUNet conditions only on `(x, t)`; **all** obstacle-awareness comes from the classifier-guidance gradient. In tight scenes, gradient guidance alone cannot reshape a globally-smooth prior sample enough. Conditioning the denoiser on a scene encoding (point cloud à la MPiNets) → classifier-free or hybrid guidance is the standard fix.

### 2.4 🟡 Representation + architecture mismatch
- **Global smoothness:** 8 control points = one degree-7 polynomial → physically cannot make the sharp turns cubby/merged_cubby require.
- **Architecture:** a 1D-conv U-Net (kernel 5, 4 down/up stages) designed for 50-waypoint sequences is applied to length-8 control-point sequences (downsampling 8→4→2→1 is degenerate). An MLP/transformer over control points likely fits better.

---

## 3. Proposed research directions (prioritized)

Each is framed as: **hypothesis → method → experiment → expected impact → cost.**

### D1. Re-calibrated noise schedule *(❌ TESTED & REFUTED — see §1b)*
- **Status:** the naive thr 0.02→0.08 recalibration *hurt* SR (66.0→53.3). The low-noise (ᾱ_T≈0.52) regime is beneficial as an implicit smoothness prior. Remaining open question: a *learned* or per-dimension SNR schedule, or even *lower* terminal noise — but deprioritized.
- **Original hypothesis (refuted):** matching ᾱ_T≈0 (proper terminal SNR) for T=64 removes the train/inference gap and raises SR across all scene types, especially constrained ones.
- **Method:** (a) rescale thresh to 0.08; (b) cosine schedule; (c) SNR-matched / "zero-terminal-SNR" schedule (Lin et al. 2024); ablate T ∈ {32,48,64,100}.
- **Experiment:** retrain each, benchmark 100/type. Compare to the 0.02-thresh baseline (this repo's current model).
- **Impact:** potentially large; cheap to test. Strong standalone paper claim ("the polynomial latent space needs its own SNR schedule").
- **Cost:** ~16 min training × ~5 configs + benchmarks.

### D2. Collision-free stitching (RRT-Connect + faithful collision) *(✅✅ VALIDATED — breaks the plateau)*
- **RESULT:** RRT-Connect + **faithful PyBullet collision checker** = **76.0%** nat-wtd vs the 62% plateau (**+14 pp**). Gains on the weak scenes: dresser 38→66, merged_cubby 24→52, cubby 72→88. Now **above paper GPD-1G (72.9%)**. The earlier "RRT only ties best-of-K" was because RRT used the loose AABB proxy; swapping in true `getContactPoints` collision checking was the unlock.
- **Implemented:** `RobotEnvironment.configs_free()`, `stitch(collision_fn=...)` threading, `run_worker.py --pybullet_collision`. Diversity (12g K=120) does not add over 1g at this scale (both 76%), though 12g is better on merged_cubby (52 vs 40).
- **Earlier ablation (AABB):** linear 54.0% → RRT 62.0% → no-stitch 62.0% (RRT repaired the harmful linear stitch but the AABB proxy capped it).
- **Next:** larger benchmark (more scenes/type) to tighten CIs; faster batched collision (sphere model) to cut RRT time; combine with best 12g/1g per scene type.
- **Why no GPDS-level gain yet (next steps):** (i) RRT's collision checker is the loose AABB intersection-volume *proxy*, not true mesh/PyBullet collision → bridges can pass the proxy but collide in reality; **fix: plug a faithful collision checker into RRT**; (ii) single-guide K=32 may lack candidate diversity (paper relies on diverse free-space coverage) → **raise K / diversity (D5)**; (iii) crude stitch-target heuristic (dist_threshold, earliest-forward).
- **Confirmed (3× independently):** linear-bridge stitching is *harmful*. The paper's GPDS gain needs faithful-collision RRT **and** diverse candidates together.
- **Hypothesis (revised):** RRT-Connect with a true collision checker + higher candidate diversity recovers most of the constrained-scene gap (paper's GPDS hits 87–92%).
- **Method:** integrate RRT-Connect (OMPL/robofin) as the bridge; OR a lightweight learned local connector; OR a short guided micro-diffusion to bridge.
- **Experiment:** ablate stitch variants on cubby/merged_cubby; report SR and added time.
- **Impact:** high on hard scenes; moderate cost.

### D3. Scene-conditioned denoiser (classifier-free / hybrid guidance)
- **Hypothesis:** a point-cloud-conditioned prior + classifier-free guidance beats guidance-only in tight scenes and reduces reliance on the 12-guide ensemble.
- **Method:** add a PointNet/PointNet++ scene encoder feeding the U-Net (FiLM conditioning); train with CFG dropout; keep cost-gradient guidance as optional hybrid term.
- **Experiment:** guidance-only vs CFG-only vs hybrid; vary #guides.
- **Impact:** potentially the largest SR ceiling lift; higher cost (architecture + retrain).

### D4. Adaptive / piecewise polynomial representation
- **Hypothesis:** allowing more control points (or piecewise segments) where curvature is needed fixes the sharp-turn failures without losing smoothness elsewhere.
- **Method:** ablate n_control ∈ {8,12,16}; or two-segment piecewise Bézier with C¹ continuity; measure SR vs speed trade-off.
- **Impact:** targeted at merged_cubby; medium cost.

### D5. Better candidate diversity for stitching *(❌ TESTED — guide-diversity does NOT help)*
- **Result:** 12-guide K=120 vs 1-guide K=32, with/without RRT, all plateau at ~60–62% (no-stitch 62.0/61.3; RRT 62.0/60.0). More guides ≠ more useful free-space coverage here. → the plateau is set by the **collision checker** (and/or model quality), not candidate count. Other diversity forms (temperature, determinantal sampling) untested but deprioritized.
- **Original hypothesis (not supported by guide-diversity):** increasing diversity improves stitch success at fixed K.
- **Method:** diversity-promoting sampling + measure coverage and stitch SR.
- **Impact:** cheap; complements D2.

---

## 4. Paper plan

**Working title:** *"Calibrated Polynomial Diffusion: Closing the Constrained-Scene Gap in Guided Motion Planning."*

**Core thesis:** GPD's speed comes from a compact polynomial latent + short diffusion chain, but that same compression introduces (i) a terminal-SNR mismatch and (ii) a smoothness/stitching bottleneck on constrained scenes. We diagnose both and propose calibrated schedules + collision-aware stitching (+ optional scene conditioning) that recover the accuracy gap while keeping the speed.

**Contributions:**
1. **Diagnosis:** first analysis showing the polynomial latent needs an SNR schedule distinct from the waypoint diffusion it's derived from; quantify the train/inference gap.
2. **Method:** calibrated/zero-terminal-SNR schedule for polynomial diffusion + collision-free stitching; (optional) scene-conditioned hybrid guidance.
3. **Empirics:** controlled ablations on MPiNets (tabletop/cubby/merged_cubby/dresser) isolating training duration vs schedule vs stitching vs conditioning; speed/accuracy Pareto vs EDMP, MPiNets, curobo.
4. **Reproduction artifact:** an open, GPU-native re-implementation with the bug/calibration fixes.

**Structure:** Intro → Related work (diffusion planners: Diffuser, MPD, EDMP, GPD; classical: RRT-Connect, CHOMP, STORM, curobo) → Background (Bernstein diffusion, guidance) → Diagnosis (Section 2 findings + evidence) → Method (D1–D3) → Experiments (ablations + Pareto) → Limitations → Conclusion.

**Target venues:** IROS / ICRA / CoRL / RA-L.

---

## 5. Immediate next experiments (concrete)
1. **Schedule sweep (D1):** retrain with thresh∈{0.04,0.08,0.12} and a cosine schedule at T=64; benchmark 100/type. (Cheapest, highest expected value.)
2. **Stitch ablation (D2):** linear-bridge vs RRT-Connect vs best-single (no stitch) on the *same* trajectories; isolates stitching's contribution.
3. **Training-duration control (this run):** 20k vs 200k at 100/type — quantifies how little duration matters (motivates structural fixes).
4. Hold all else fixed (12 guides, K=120) so each ablation is clean.

_Last updated: 200k re-benchmark complete. Control experiment confirms training duration is not the bottleneck (+2.8 pp overall from 10× training; constrained-scene gap intact)._

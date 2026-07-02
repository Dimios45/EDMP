# Improving Guided Polynomial Diffusion (GPD): Diagnosis, Research Directions, and Paper Plan

**Base:** GPD ([arXiv:2501.18229](https://arxiv.org/html/2501.18229v1)) implemented inside the EDMP codebase.
**Hardware:** RTX 4090. **Robot:** Franka Panda (7-DoF). **Benchmark:** MPiNets scenes (tabletop / cubby / merged_cubby / dresser).

---

## 1. Reproduction status (our results)

We trained `GPDModel64_N8` (TemporalUNet, dims (32,64,128,256), ~3.95M params, T=64, 8 Bernstein control points) on the 6.54M-trajectory MPiNets set (global + hybrid solutions projected to Bernstein control points), then benchmarked at **100 scenes/type (400 scenes)** with 12 guides × 10 samples + linear-bridge stitching.

| Scene | 20k-step model | 200k-step model | Δ (10× train) | BLOG GPD | Paper GPD-stitched |
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

We targeted **one** paper baseline — **single-guide GPD** (single guide, K=32 candidates, best-of-K selection, **no stitching**) — to validate the core method, then tested the schedule-recalibration schedule idea as a controlled ablation. All on hybrid-solvable, natural-distribution, reported as natural-weighted SR.

| Run | Config | nat-wtd SR | per-type (tab/cub/merg/dres) |
|---|---|---|---|
| **Exp 1** (300 scenes) | single-guide GPD, 200k, thr 0.02 | **62.3%** | 96 / 56 / 44 / 41 |
| 2a (150 scenes) | single-guide GPD, 20k, thr 0.02 | 66.0%* | 100 / 72 / 28 / 48 |
| 2b (150 scenes) | single-guide GPD, 20k, **thr 0.08 (schedule-recalibration)** | 53.3% | 84 / 44 / 16 / 46 |
| (prior) | **12-guide + linear stitch**, 200k | ≈53% | 98 / 32 / 24 / 33 |
| Paper | single-guide GPD | **72.9%** | — |

*2a used a smaller/easier 150-scene subset than Exp 1's 300; not directly comparable to Exp 1, but 2a vs 2b *are* same-scene and so a clean schedule-recalibration ablation.

**Findings:**
1. **single-guide GPD reproduced to within ~10 pp** (Exp 1 62.3% vs paper 72.9%). The previously-alarming ~38 pp "gap" was largely a **configuration artifact**: our headline runs used 12 guides + linear-bridge stitching (≈53%), which *underperforms* the paper's simple single-guide best-of-K. Residual ~10 pp likely = guide-hyperparameter / K / success-criterion / guide-choice differences.
2. **schedule recalibration is REFUTED.** Same scenes, matched 20k budget: thr 0.02 → 0.08 *lowered* SR on all 4 types (66.0 → 53.3 nat-wtd) despite *lower* training loss (0.53 → 0.43). Mechanism: ᾱ_T≈0.52 keeps the reverse process near-identity → lightly polishes smooth, demonstration-like trajectories. Raising terminal noise injects more reverse-process variance → less-smooth, more-colliding paths. **The "miscalibration" is an implicit smoothness prior; ε-loss is a poor proxy for planning success.**
3. **Linear-bridge stitching is harmful**, not just suboptimal: single-guide best-of-K (no stitch) **beats** 12-guide + linear stitch. → confirms collision-aware stitching from a second angle.

**Reprioritization:** schedule-recalibration ↓ (drop the naive schedule fix; the low-noise regime is good). **collision-aware stitching ↑↑** (fix or remove stitching) is now the top lever. scene-conditioning and polynomial-capacity (representation) unchanged.

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

### Re-calibrated noise schedule *(❌ TESTED & REFUTED — see §1b)*
- **Status:** the naive thr 0.02→0.08 recalibration *hurt* SR (66.0→53.3). The low-noise (ᾱ_T≈0.52) regime is beneficial as an implicit smoothness prior. Remaining open question: a *learned* or per-dimension SNR schedule, or even *lower* terminal noise — but deprioritized.
- **Original hypothesis (refuted):** matching ᾱ_T≈0 (proper terminal SNR) for T=64 removes the train/inference gap and raises SR across all scene types, especially constrained ones.
- **Method:** (a) rescale thresh to 0.08; (b) cosine schedule; (c) SNR-matched / "zero-terminal-SNR" schedule (Lin et al. 2024); ablate T ∈ {32,48,64,100}.
- **Experiment:** retrain each, benchmark 100/type. Compare to the 0.02-thresh baseline (this repo's current model).
- **Impact:** potentially large; cheap to test. Strong standalone paper claim ("the polynomial latent space needs its own SNR schedule").
- **Cost:** ~16 min training × ~5 configs + benchmarks.

### Collision-free stitching (RRT-Connect + faithful collision) *(✅✅ VALIDATED — breaks the plateau)*
- **RESULT:** RRT-Connect + **faithful PyBullet collision checker** = **76.0%** nat-wtd vs the 62% plateau (**+14 pp**). Gains on the weak scenes: dresser 38→66, merged_cubby 24→52, cubby 72→88. Now **above paper single-guide GPD (72.9%)**. The earlier "RRT only ties best-of-K" was because RRT used the loose AABB proxy; swapping in true `getContactPoints` collision checking was the unlock.
- **Implemented:** `RobotEnvironment.configs_free()`, `stitch(collision_fn=...)` threading, `run_worker.py --pybullet_collision`. Diversity (12g K=120) does not add over 1g at this scale (both 76%), though 12g is better on merged_cubby (52 vs 40).
- **Earlier ablation (AABB):** linear 54.0% → RRT 62.0% → no-stitch 62.0% (RRT repaired the harmful linear stitch but the AABB proxy capped it).
- **Next:** larger benchmark (more scenes/type) to tighten CIs; faster batched collision (sphere model) to cut RRT time; combine with best 12g/1g per scene type.
- **Why no GPD-stitched-level gain yet (next steps):** (i) RRT's collision checker is the loose AABB intersection-volume *proxy*, not true mesh/PyBullet collision → bridges can pass the proxy but collide in reality; **fix: plug a faithful collision checker into RRT**; (ii) single-guide K=32 may lack candidate diversity (paper relies on diverse free-space coverage) → **raise K / diversity (guide-diversity)**; (iii) crude stitch-target heuristic (dist_threshold, earliest-forward).
- **Confirmed (3× independently):** linear-bridge stitching is *harmful*. The paper's GPD-stitched gain needs faithful-collision RRT **and** diverse candidates together.
- **Hypothesis (revised):** RRT-Connect with a true collision checker + higher candidate diversity recovers most of the constrained-scene gap (paper's GPD-stitched hits 87–92%).
- **Method:** integrate RRT-Connect (OMPL/robofin) as the bridge; OR a lightweight learned local connector; OR a short guided micro-diffusion to bridge.
- **Experiment:** ablate stitch variants on cubby/merged_cubby; report SR and added time.
- **Impact:** high on hard scenes; moderate cost.

### Scene-conditioned denoiser (classifier-free / hybrid guidance) *(❌ REFUTED — conditioning does not raise the ceiling)*
- **Hypothesis:** a scene-conditioned prior + CFG beats guidance-only in tight scenes.
- **Method (built):** masked DeepSets obstacle-set encoder → scene emb ADDED to time emb (flows through existing FiLM, no block changes) + learned null embedding for CFG; trained 60k with 15% scene-dropout. `gpd/{conditional_unet,preprocess_scenes}.py`, `train_gpd_cond.py`, `run_worker.py --conditional --cfg_weight`. Loss **0.502 < uncond 0.52** (scene info *did* lower ε-loss).
- **❌ RESULT (GPD-stitched, 100 sc/type hybrid):**
  | config | overall | tab/cub/merg/dres |
  |---|---|---|
  | A uncond + guidance (baseline) | 71.8 | 99/68/61/59 |
  | B cond + guidance (hybrid, w=1) | 72.5 | 100/68/60/62 |
  | C **cond, NO guidance (CFG-only)** | **59.8** | 97/46/32/64 |
  | D **uncond, NO guidance (prior-only)** | **61.0** | 98/46/37/63 |
  - Hybrid (B) ≈ baseline (A): wash. Decisive: **C ≈ D** — the conditioned prior plans *no better than the scene-blind prior* even with no guidance to mask it. Guidance = +11pp (A−D); conditioning ≈ 0pp.
- **Confound ruled out:** inference scene features verified in-distribution vs train (centers/sizes overlap, quat convention matches w-first). Conditioning is genuinely useless for planning SR, not broken.
- **Verdict:** FOURTH instance of ε-loss ⊥ planning-SR decoupling (schedule-recalibration, polynomial-capacity, smoothness-penalty, scene-conditioning). The prior already has all it needs; scene info enters effectively via guidance, and adding it to the prior changes nothing.

### Adaptive / piecewise polynomial representation *(❌ NAIVE VARIANT REFUTED — capacity backfires; see result below)*
- **❌ RESULT (N16 vs N8, both @20k, GPD-stitched config, 100 sc/type hybrid):** overall **71.8% → 57.2% (−14.6pp)**; tabletop 99→98, cubby 68→56, **merged_cubby 61→28 (−33!)**, dresser 59→47. N16 worse everywhere, catastrophic on the most-constrained scene — the *opposite* of the hypothesis. Validity anchor: N8@20k GPD-stitched (71.8%) ≈ N8@200k replication (70.7%) → budget fair.
- **Why it backfired:** the reconstruction diagnostic (below) proved headroom exists *in the demos*, but the diffusion prior cannot exploit it — extra control points produce jaggier, more-colliding samples; **losing the low-degree smoothness regularizer dominates the representational gain.** Same lesson as schedule-recalibration: GPD's compactness is an implicit smoothness prior, and nominal loss / reconstruction-fidelity are anti-correlated with planning SR.
- **Confound RESOLVED (N16→100k, loss 0.476 ≈ N8@200k):** N16@100k = **65.5%** (tab 94 / cub 56 / merg 47 / dres 65). Under-training was partly real (+8.3pp from 5× training; merged_cubby recovers 28→47), but at matched convergence N16 is **still −6.3pp below N8**. Net loss, with a redistribution: N16 *beats* N8 on the sharp-turn scenes (dresser 59→65) yet *loses* on tabletop/cubby — and costs 5× training.
- **Smoothness-constrained variant (control-point curvature penalty, λ-sweep on N16@20k):** peak **59.2% @ λ=0.03** (+3.4pp over λ=0); λ=0.12 collapses (34.6%). Recovers only a fraction; nowhere near N8.
- **FINAL VERDICT (3 experiments agree):** added representational capacity is a net loss for planning SR despite 4× lower reconstruction error + equal train loss. Reconstruction fidelity is **decoupled from** planning SR. With schedule-recalibration (schedule) this triangulates the thesis: **GPD's compactness is an implicit, well-conditioned regularizer; the binding constraint is INFORMATION (environment conditioning), not capacity.** → scene-conditioning is the sole remaining ceiling-raiser.
- **Hypothesis:** allowing more control points (or piecewise segments) where curvature is needed fixes the sharp-turn failures without losing smoothness elsewhere.
- **⭐ GO-SIGNAL (reconstruction-error diagnostic, no training needed).** Bernstein fit error (worst joint × worst waypoint, in degrees) on 20k expert demos vs n_control:
  | demos | n=8 | n=16 | n=20 |
  |---|---|---|---|
  | global (easy) mean / %>2° | 0.96° / 12.5% | 0.15° / 0% | 0.09° / 0% |
  | **hybrid (hard) mean / %>2° / %>5°** | **5.47° / 89.8% / 44.3%** | 1.43° / 19.3% / 0.3% | 0.99° / 5.3% / 0% |
  → **degree-7 global Bézier discards 5–17° of joint motion on the constrained (hybrid) demos** — exactly the cubby/merged_cubby/dresser scenes where GPD fails. The model inherits a *representational* ceiling from over-smoothed training control points; no denoiser/guidance can recover it. This is the mechanism behind the constrained-scene gap and a standalone diagnostic figure.
- **Method:** n=16 (clean 4-stage UNet downsample 16→8→4→2; n=12/20 don't divide). Re-preprocessed combined data → `gpd_train_combined_n16.hdf5` (6.54M, 7, 16). Training `GPDModel64_N16` at 20k steps = **same budget as the existing `GPDModel64_N8_ckpt20k`** → clean single-variable ablation. `run_worker.py --n_control 16` added.
- **Experiment:** N16 vs N8(@20k) on hybrid, GPD-stitched config (1 guide + faithful-PyBullet RRT stitch), per-type SR — watch cubby/merged_cubby/dresser.
- **Risk to test:** more control points = freedom to capture sharp turns (good) OR jagged colliding prior samples (bad — the low-noise schedule's smoothness prior helped). Benchmark decides.
- **Impact:** targeted at merged_cubby/dresser; cheap (preprocess done, ~40min train).

### Better candidate diversity for stitching *(❌ TESTED — guide-diversity does NOT help)*
- **Result:** 12-guide K=120 vs 1-guide K=32, with/without RRT, all plateau at ~60–62% (no-stitch 62.0/61.3; RRT 62.0/60.0). More guides ≠ more useful free-space coverage here. → the plateau is set by the **collision checker** (and/or model quality), not candidate count. Other diversity forms (temperature, determinantal sampling) untested but deprioritized.
- **Original hypothesis (not supported by guide-diversity):** increasing diversity improves stitch success at fixed K.
- **Method:** diversity-promoting sampling + measure coverage and stitch SR.
- **Impact:** cheap; complements collision-aware stitching.

---

### Continuous-feasibility trajectory-optimization repair *(✅✅✅ VALIDATED — +10.2pp, breaks the plateau)*
- **Motivation (failure-mode analysis, 200 hybrid scenes):** the dominant baseline failure is **dynamic_only = 15.5%** (path collision-free at every waypoint, but execution collides — the robot *sweeps through* obstacles between the 50 waypoints; the stitcher only point-checks). path_collision 8.0%, goal/start-IK-infeasible 2.5%. → the bottleneck is *continuous* feasibility, not the prior.
- **Literature (June 2026):** dominant diffusion-MP theme = trajectory-optimization feasibility-repair warm-started from the diffusion seed (PRESTO ICRA'25, DGD, DRAFTO, MPD). Untried here; aligns with the thesis.
- **Method (`gpd/refine.py` `trajopt_refine`):** Adam on interior waypoints minimizing the guide's **differentiable** intersection-volume cost at **waypoints + edge-midpoints** + smoothness, endpoints pinned, joint-limit clamp; early-stop on a **densified** faithful (PyBullet) collision check; return a densified trajectory (×3) for overshoot-free execution. Reuses the existing differentiable guide cost — no new model/training.
- **✅ RESULT (100 sc/type hybrid, GPD-stitched):**
  | config | overall | tab/cub/merg/dres |
  |---|---|---|
  | base | 72.8 | 99/70/55/67 |
  | densify-only | 73.0 | 100/68/56/68 |
  | **trajopt + densify** | **83.0** | 99/**90**/66/77 |
  - **Densify-only ≈ base** → dynamic failures are genuine swept-collision, not PD overshoot.
  - **Trajopt repair = +10.2pp** (cubby +20, dresser +10, merged +11). The **first** lever besides faithful stitching to move SR, and it breaks the ~72% plateau that resisted schedule/conditioning/capacity. Cost: +~1.8s/scene, only on failing scenes (early-stops when edge-free).
- **Next:** extend to global/both datasets + larger N; ablate out_densify / midpoint cost / iters; report time/accuracy Pareto.

## 4. Paper plan

**Working title:** *"The Prior Is Not the Bottleneck: Where Success Actually Comes From in Guided Diffusion Motion Planning."*

**Core thesis (5 controlled experiments):** In guided polynomial diffusion planners, planning success is bottlenecked by the **inference-time feasibility machinery** (cost-gradient guidance + collision-checked stitching), **not by the generative prior**. The ~72% balanced-hybrid ceiling is *invariant* to every prior-side intervention we tried — noise schedule, polynomial capacity, smoothness penalty, and scene conditioning — several of which *lower the training loss while leaving (or reducing) planning SR*. The only lever that moves the ceiling is improving the feasibility checker: faithful-collision RRT stitching (+14pp). **Nominal generative metrics (ε-loss, demo-reconstruction fidelity) are decoupled from — even anti-correlated with — planning SR.** This is a contrarian message for a field that keeps improving priors.

**Contributions:**
1. **A decoupling result:** four independent prior improvements (schedule recalibration, ↑capacity, smoothness, scene-conditioning) each lower ε-loss / reconstruction error yet do **not** raise planning SR — with mechanism (compression acts as a well-conditioned regularizer; guidance already supplies scene info, making conditioning redundant).
2. **A failure-mode diagnosis:** the dominant residual failure is *continuous/swept* infeasibility (collision between waypoints), not static or prior quality — quantified by a goal/path/dynamic breakdown.
3. **Two positive levers, both on the feasibility machinery:** (a) faithful-collision RRT stitching, +14pp over the AABB-proxy plateau (the proxy certifies colliding bridges as free); (b) **continuous-feasibility trajectory-optimization repair (PRESTO/DRAFTO-style), +10.2pp (72.8→83.0)**, warm-started from the diffusion seed, minimizing collision at waypoints+midpoints — breaks the plateau every prior-side intervention could not.
4. **A diagnostic methodology + open artifact:** controlled single-variable ablations on MPiNets separating prior quality from feasibility machinery, plus a GPU-native GPD re-implementation with all ablations, the conditioning extension, the stitching fix, and the trajopt-repair module.

**Structure:** Intro → Related work (diffusion planners: Diffuser, MPD, EDMP, GPD; classical: RRT-Connect, CHOMP, STORM, curobo) → Background (Bernstein diffusion, guidance) → Diagnosis (Section 2 findings + evidence) → Method (the prior interventions) → Experiments (ablations + Pareto) → Limitations → Conclusion.

**Target venues:** IROS / ICRA / CoRL / RA-L.

---

## 5. Immediate next experiments (concrete)
1. **Schedule sweep (schedule-recalibration):** retrain with thresh∈{0.04,0.08,0.12} and a cosine schedule at T=64; benchmark 100/type. (Cheapest, highest expected value.)
2. **Stitch ablation (collision-aware stitching):** linear-bridge vs RRT-Connect vs best-single (no stitch) on the *same* trajectories; isolates stitching's contribution.
3. **Training-duration control (this run):** 20k vs 200k at 100/type — quantifies how little duration matters (motivates structural fixes).
4. Hold all else fixed (12 guides, K=120) so each ablation is clean.

_Last updated: 200k re-benchmark complete. Control experiment confirms training duration is not the bottleneck (+2.8 pp overall from 10× training; constrained-scene gap intact)._

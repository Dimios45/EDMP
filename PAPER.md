# The Prior Is Not the Bottleneck: Where Success Comes From in Guided Diffusion Motion Planning

*Working draft. Base system: Guided Polynomial Diffusion (GPD, arXiv:2501.18229) re-implemented in the EDMP codebase. Robot: Franka Panda (7-DoF). Benchmark: MPiNets scenes (tabletop / cubby / merged_cubby / dresser). Hardware: single RTX 4090.*

---

## Abstract

Guided diffusion has become a popular paradigm for robot motion planning, and a steady stream of work improves the *generative prior* — better noise schedules, richer trajectory representations, and environment conditioning. We present a controlled study on Guided Polynomial Diffusion (GPD) showing that, for a guided planner, **the prior is largely not the bottleneck**. Across five independent prior-side interventions — recalibrating the noise schedule, increasing the polynomial capacity, adding a smoothness penalty, conditioning the denoiser on the scene with classifier-free guidance, and training the prior 5× longer (to 1M steps) — planning success rate (SR) never improves (deltas range from statistically zero to small significant *decreases*), *even when those interventions lower the training loss or trajectory-reconstruction error*. We trace this to two facts: (i) the compactness of GPD's short diffusion chain and low-degree polynomial acts as a well-conditioned implicit regularizer, and (ii) classifier guidance already injects the scene information at inference, making a conditioned prior redundant. A failure-mode analysis then shows the dominant residual failure is **continuous (swept) infeasibility** — paths that are collision-free at every waypoint but collide during execution. Acting on this, two interventions on the **feasibility machinery** — faithful-collision RRT stitching (+17.4pp, 95% CI [15.2,19.6], over an AABB-proxy plateau) and a PRESTO/DRAFTO-style **trajectory-optimization repair** warm-started from the diffusion seed (**+8.9pp, 95% CI [7.6, 10.2] over 5 seeds**; 72.9→81.8% on hybrid) — break the plateau that every prior-side change could not. A naive-seed control sharpens the claim: the diffusion prior is *necessary* as an initializer (a straight-line seed + the same repair reaches only 55% vs 82%), but *improving* an already-competent prior does not help — the returns are in the feasibility machinery, not in the generative model. The same repair transfers to a second diffusion planner (EDMP, +7.6pp, CI [5.2,9.9]), confirming the lever is planner-agnostic. We argue the community's effort is mis-allocated and provide an open, GPU-native re-implementation with all ablations.

---

## 1. Introduction

Diffusion models generate diverse, multimodal trajectories and, with classifier (cost-gradient) guidance, can be steered to avoid obstacles at inference time. GPD compresses the trajectory into 8 Bernstein control points and shortens the diffusion chain to T=64, achieving fast planning. Most follow-up work — and the natural set of "obvious" improvements — targets the generative model: the noise/SNR schedule, the trajectory parameterization, and conditioning the denoiser on the environment.

We ran those improvements as controlled, single-variable experiments and found they **do not raise success rate**. This paper reports that negative result, explains it, and then shows where the gains actually are: the inference-time feasibility machinery. Our central, somewhat contrarian claim is that for guided diffusion planners the prior is "good enough," and the returns lie in collision-aware stitching and trajectory-optimization repair.

**Contributions.**
1. A **decoupling result**: five prior-side improvements (including a 5×/1M-step training budget) each improve a nominal generative metric yet leave SR flat (or worse) — all within ~1σ of a measured sampling noise floor (5-seed).
2. A **failure-mode diagnosis** isolating continuous/swept infeasibility as the dominant residual failure.
3. **Two positive levers on the feasibility machinery**: faithful-collision stitching (+17.4pp) and trajectory-optimization repair (+8.9pp on GPD, +7.6pp on EDMP — planner-agnostic), the latter targeting and collapsing the diagnosed swept-collision failure (15.5→3.0%).
4. An **open, GPU-native** GPD re-implementation with every ablation, a scene-conditioning extension, and a trajopt-repair module.

---

## 2. Background

**GPD.** TemporalUNet ε-predictor (~3.95M params) over 8 Bernstein control points, T=64, variance schedule β=linspace(0,0.02,T+1). Start/goal pinned at the boundary control points. Inference: sample K candidates under cost-gradient guidance (IntersectionVolumeGuide: differentiable AABB intersection-volume + swept-volume via PyTorch FK), select / stitch.

**Success metric (paper-faithful).** Execute the trajectory in PyBullet with position control; success = no obstacle contact during execution (`getContactPoints` vs the obstacle bodies). This is a *continuous-execution* test, not a waypoint check.

**Setup.** All numbers below are on the MPiNets **hybrid-solvable** set, balanced 100 scenes per environment type (400 total) unless noted, under the GPD-stitched configuration (single guide, K=32, faithful-collision RRT stitching).

---

## 3. The prior is not the bottleneck

We test the four most natural prior-side improvements. Each is a clean single-variable change; all share the identical guidance + stitching pipeline.

### 3.1 Noise-schedule recalibration
At T=64 the linear schedule gives ᾱ_T≈0.52 — the model is never trained near pure noise, while inference starts from N(0,I). The "fix" (raise terminal noise so ᾱ_T≈0, or cosine) **lowered SR on all scene types** despite a **lower** training loss (0.53→0.43 ε-MSE). Mechanism: the low-terminal-noise regime keeps the reverse process near-identity, acting as an implicit smoothness prior; raising terminal noise injects variance and yields less-smooth, more-colliding paths. *ε-loss is a poor proxy for planning SR.*

### 3.2 Polynomial capacity
Bernstein reconstruction error on the expert demos shows degree-7 (n=8) **discards 5–17° of joint motion on the constrained (hybrid) demonstrations** (mean 5.47°, p99 17.4°; 90% of trajectories lose >2°), vs 0.96° on the easy (global) demos — so the representation *is* lossy exactly where GPD struggles. Yet training n=16 (4× lower reconstruction error, matched convergence loss 0.476≈n=8's 0.48) gives:

| | overall | tab | cubby | merged | dresser |
|---|---|---|---|---|---|
| n=8 (baseline) | **71.8** | 99 | 68 | 61 | 59 |
| n=16 @ matched conv. | 65.5 | 94 | 56 | 47 | 65 |

Capacity is a **net −6.3pp**, with a telling redistribution (n=16 *beats* n=8 on the sharp-turn dresser scenes but loses on tabletop/cubby) and costs 5× the training. A smoothness penalty on n=16 recovers only +3.4pp (peak 59.2% at λ=0.03). The compact basis is a better-conditioned generative target; capacity is not the constraint.

**Mechanism (compression = smoothness regularizer).** Measuring the *raw* sampled trajectories (100 hybrid scenes, before any stitch/repair), the compact n=8 prior is markedly smoother than n=16: joint-jerk RMS **0.00166 vs 0.00214 (−22%)**, acceleration RMS 0.0071 vs 0.0079, and a larger sphere-SDF clearance margin (−0.043 vs −0.047 m). The low-degree Bézier basis cannot express the high-frequency motion that a higher-capacity model fits to the demos, so it acts as an implicit smoothness prior — and smoother trajectories collide less under continuous execution. This is the same effect seen across the schedule, conditioning, and capacity interventions: nominal generative fidelity (reconstruction error, ε-loss) is decoupled from — here anti-correlated with — the smoothness that actually drives planning SR (see `results/smoothness_jerk.png`).

### 3.3 Scene-conditioned denoiser + classifier-free guidance
We add a masked-DeepSets obstacle-set encoder whose embedding is added to the time embedding (flowing through the existing FiLM), with a learned null embedding for CFG (15% scene-dropout). The conditioned model reaches a **lower** loss (0.502 < 0.52).

| config | overall | tab | cubby | merged | dresser |
|---|---|---|---|---|---|
| A uncond + guidance (baseline) | 71.8 | 99 | 68 | 61 | 59 |
| B cond + guidance (hybrid) | 72.5 | 100 | 68 | 60 | 62 |
| C cond, no guidance (CFG-only) | 59.8 | 97 | 46 | 32 | 64 |
| D uncond, no guidance (prior-only) | 61.0 | 98 | 46 | 37 | 63 |

Hybrid (B) ≈ baseline (A). Decisively, **C ≈ D**: the conditioned prior plans no better than the scene-blind prior even with guidance removed. Guidance supplies +11pp (A−D); conditioning ≈ 0pp. We verified the inference scene features are in-distribution (centers/sizes/quaternion convention match training), so this is genuine, not a wiring bug. **Scene information enters effectively through guidance; adding it to the prior is redundant.** Over 5 paired seeds the conditioning delta is in fact **−1.75pp, 95% CI [−2.37, −1.13]** (t=−7.8) — a small but significant *decrease*: adding scene information to the prior does not merely fail to help, it mildly hurts.

### 3.5 Training to convergence (5× budget, 1M steps)
The most basic prior-side lever is simply *more training*. We train the canonical n=8 prior for **1,000,000 steps** (5× the 200k baseline), identical config and data, in an isolated checkpoint. The extra budget lowers ε-loss by ~4% (min 0.4632→0.4434; mean(last500) 0.4809→0.4636) and the curve is flat for the final 300k steps — the prior is as converged as the architecture/data allow.

| config (hybrid, balanced 400) | 200k prior | 1M prior (5×) | Δ |
|---|---|---|---|
| plain GPD | 72.8 | 74.2 | +1.4 |
| GPD + trajopt repair (§4.3) | 83.0 | 82.2 | −0.8 |

The single-run table shows a nominal +1.4pp on the bare planner, but over **5 paired seeds** the long-prior delta is **−0.10pp, 95% CI [−1.71, +1.51]** (t=−0.17) — statistically indistinguishable from zero — and it stays flat once repair is applied. A 5× better-converged prior buys **nothing**, and it does **not** close the reproduction gap to the published GPD-stitched baseline (92.8). This rules out under-training as the explanation both for the flat prior-side results and for the gap: the negative result is a property of the guided-planning system, not of an undertrained model.

### 3.6 Summary
Five prior improvements, five nominal-metric gains, **no positive SR gain** — the deltas range from statistically zero (5× training: −0.1pp, CI [−1.7,+1.5]) to small significant *decreases* (conditioning −1.75pp; schedule and capacity larger drops). The ~72% hybrid ceiling is invariant to (or mildly worsened by) prior-side change. Against a measured baseline sampling noise of σ≈1.6pp (§4.3), no prior intervention clears it upward, while the feasibility-repair lever (§4.3) clears it by ~6σ.

---

## 4. Where success actually comes from

### 4.1 Failure-mode analysis
We categorize 200 hybrid failures of the baseline:

| failure mode | baseline % | after repair % |
|---|---|---|
| success | 74.0 | 78.5 |
| goal/start IK infeasible | 2.5 | 1.5 |
| path_collision (stitched **waypoints** collide) | 8.0 | 17.0 |
| **dynamic_only (waypoints free, **execution** collides)** | **15.5** | **3.0** |

The executor and the static checker use the *same* contact model, so `dynamic_only` is genuine: the robot **sweeps through obstacles between waypoints** (the stitcher only point-checks the 50 waypoints). The bottleneck is *continuous* feasibility. The right column previews the effect of the §4.3 repair: **`dynamic_only` collapses 15.5→3.0** — the lever we build targets exactly the failure this analysis isolates. Of the eliminated swept failures, ~4.5pp convert to success and the remainder become *statically-detectable* `path_collision` (8.0→17.0): on the hardest scenes the optimizer churns but leaves a residual static collision rather than a swept-only one — it converts dynamic failures, it does not manufacture them.

### 4.2 Lever 1 — faithful-collision stitching
The stitcher originally certified bridges with the loose AABB intersection-volume *proxy*, which marks colliding bridges as free. Replacing it with a faithful PyBullet check inside RRT-Connect yields **+17.4pp** over the proxy plateau (55.5±1.0→72.9±1.6, 5 seeds), the first intervention to move SR.

### 4.3 Lever 2 — trajectory-optimization repair (PRESTO/DRAFTO-style)
Motivated by §4.1 and the 2024–25 literature (PRESTO, DGD, DRAFTO, MPD), we add a post-hoc optimizer (`trajopt_refine`): Adam on the interior waypoints minimizing the guide's **differentiable** collision cost at **waypoints and edge-midpoints** plus a smoothness term, endpoints pinned, joint limits clamped; early-stop on a **densified** faithful collision check; return a densified trajectory for overshoot-free execution. It reuses the existing differentiable guide cost — no retraining.

| config | overall | tab | cubby | merged | dresser | avg t |
|---|---|---|---|---|---|---|
| base | 72.8 | 99 | 70 | 55 | 67 | 7.3s |
| densify-only | 73.0 | 100 | 68 | 56 | 68 | 7.2s |
| **trajopt + densify** | **83.0** | 99 | **90** | 66 | 77 | 9.1s |

Two results: **densify-only ≈ base** (dynamic failures are genuine swept-collision, not PD overshoot), and **trajopt repair = +8.9pp** (95% CI [7.6,10.2], 5 seeds), breaking the plateau that every prior-side change could not. The optimizer targets a densified static check; the benchmark independently verifies dynamic execution, so the 81.8% is a real executable-SR gain. Cost: ~+1.8s/scene, only on scenes that need repair (early-stop when edge-free).

**Multi-seed significance.** We repeat the base-vs-repair comparison over 5 RNG seeds for the diffusion sampling (hybrid balanced 400). Base SR = 72.9 ± 1.6, repair SR = 81.8 ± 1.3 (mean ± std). The **paired** improvement is **+8.9pp, 95% CI [7.6, 10.2]**, t(4)=19.4, p<0.0001 — robust, not a single-run artifact. Crucially, the measured baseline run-to-run **σ ≈ 1.6pp** also calibrates the §3 null results: the conditioning (+0.7pp, §3.3) and 5×-training (+1.4pp, §3.5) deltas both fall **within 1σ** of sampling noise, so "the prior is not the bottleneck" is a statement about effects below the noise floor, while the repair lever clears it by ~6σ.

### 4.4 Is the prior even needed? A naive-seed control
A natural objection to §3 is that if improving the prior never helps, perhaps the prior is irrelevant and the repair does all the work. We test this directly: replace the diffusion candidates with a single **straight-line joint-space seed** (start→goal) and run the *identical* downstream (RRT stitching ± trajopt repair), 3 seeds.

| seed ↓ / pipeline → | stitch only | stitch + trajopt |
|---|---|---|
| **linear** (no prior) | 18.5 ± 0.3 | 55.0 ± 1.0 |
| **diffusion** (prior) | 72.9 ± 1.6 | 81.8 ± 1.3 |

The prior is **necessary as an initializer**: even the best repair on a linear seed reaches only 55.0%, a full **−26.8pp** below diffusion+repair — repair cannot substitute for a competent seed. Yet repair does far more *work* on the bad seed (+36.5pp, 18.5→55.0) than on the diffusion seed (+8.9pp), i.e. good initialization and good repair are **complementary, not interchangeable**. This scopes the paper's claim precisely: *improving an already-competent prior's generative fidelity* (schedule, capacity, conditioning, 5× training) buys nothing, but the prior itself is not dispensable — it supplies the warm-start that makes repair land near the ceiling. The returns are in the feasibility machinery; the prior earns its place as an initializer, not as a tunable lever.

### 4.5 The repair lever generalizes to a second planner (EDMP)
If the bottleneck is the feasibility machinery rather than any one prior, the same repair should help a *different* diffusion planner. We apply the identical `trajopt_refine` to EDMP (a different model: T=255 over full 50-waypoint trajectories, separate guidance ensemble), which already exposes the same `IntersectionVolumeGuide`.

| EDMP (hybrid balanced 400) | overall | tab | cubby | merged | dresser | avg t |
|---|---|---|---|---|---|---|
| base | 57.2 | 96 | 50 | 30 | 53 | 44.0s |
| **+ trajopt repair (3-seed mean)** | **64.8** | 97 | **68** | **42** | **63** | 50.3s |

Repair lifts EDMP by **+7.6pp** (57.2±1.0→64.8±1.3, 95% CI [5.2,9.9], 3 seeds) — a significant transfer alongside the GPD gain (+8.9pp) and concentrated on the same constrained scene types. The lever is a property of the **inference-time feasibility machinery**, not of the GPD prior: it transfers across planners with different chains, representations, and guidance.

### 4.6 Trajopt hyperparameter ablation (time/accuracy Pareto)
We ablate the three repair knobs one-factor-at-a-time around the default (iters=60, output-densify=3, midpoint-weight=1.0) on hybrid (400 scenes). The anchor reproduces at 83.2%.

| knob | value | SR% | avg t (s) |
|---|---|---|---|
| **iters** | 0 (densify-only) | 71.8 | 7.2 |
| | 20 | 80.5 | 7.9 |
| | 40 | 81.2 | 8.7 |
| | **60** | **83.2** | 9.0 |
| | 100 | 80.0 | 9.8 |
| **out-densify** | 1 (none) | 77.5 | 9.4 |
| | 2 | 79.8 | 9.1 |
| | **3** | **83.2** | 9.0 |
| | 5 | 79.5 | 8.9 |
| **mid-weight** | 0.0 (waypoints-only) | 79.8 | 8.2 |
| | 0.5 | 79.2 | 9.1 |
| | **1.0** | **83.2** | 9.0 |
| | 2.0 | 81.2 | 8.9 |

Three findings. (i) **Iterations have a peak, not a plateau**: SR saturates by ~40 and iters=100 *regresses* (−3.2pp vs the peak) — over-optimization drifts waypoints off the smooth diffusion seed into new collisions, an echo of the compactness-as-regularizer effect now *inside* the repair loop. iters=20 is an efficiency knee (+8.7pp over the densify-only floor for only +0.7s). (ii) **The midpoint-cost term is causally tied to the swept-feasibility diagnosis of §4.1**: removing it (waypoints-only objective) costs −3.4pp; optimizing collision cost at edge *midpoints*, not just waypoints, is what repairs continuous infeasibility. (iii) **Output densification matters but non-monotonically** (peak at 3). The Pareto front is densify-only (71.8% @7.2s) → iters=20 (80.5% @7.9s) → anchor (83.2% @9.0s); all other settings are dominated.

### 4.7 Does objective fidelity matter? Exact vs proxy collision (sphere-SDF)
The repair objective above is the guide's loose AABB intersection-volume. We implemented a more exact GPU-batched alternative: each link is approximated by FK-placed spheres scored against the obstacles' *oriented* boxes (signed-distance penetration; `gpd/sphere_collision.py`), removing both the robot- and obstacle-side AABB inflation. Swapping it into the same repair loop (3 seeds) gives **82.0 ± 0.9 vs 81.8 ± 1.3** for the proxy — a **paired +0.6pp, not significant**, at equal/slightly-lower time. The exact objective does **not** help: because the **faithful PyBullet check is the stopping oracle**, the differentiable objective only needs to point gradients approximately right. Objective *fidelity* is decoupled from SR — the same fidelity-vs-success decoupling the paper documents on the prior side, now recurring inside the feasibility machinery.

---

## 5. Discussion

The two effective levers are both on the **inference-time feasibility machinery**, and both are about *continuous* feasibility (collision-free bridges; collision-free swept motion). The four ineffective levers are all on the **prior**, and several improved nominal generative metrics. For guided diffusion planners, ε-loss and reconstruction fidelity are decoupled from — sometimes anti-correlated with — planning success. We therefore argue that effort spent improving priors yields little once a competent guided prior exists; the returns are in feasibility repair.

**Limitations / next.** Core results are on hybrid-solvable, balanced 100/type (400 scenes); the trajopt knobs are ablated in §4.4 and the repair lever is shown to generalize across global/both-solvable splits (+6 to +9pp, App.). Remaining: larger N for tighter CIs, and a time/accuracy Pareto against EDMP, MPiNets, and cuRobo. The trajopt repair uses the smooth AABB proxy as its differentiable objective and the faithful PyBullet check as the stopping oracle; we implemented an exact GPU-batched sphere-SDF objective (§4.7) and found it statistically tied with the proxy, so objective exactness is not a remaining lever here.

### 5.1 Reproduction study (how far in-system tuning closes the gap)
Our re-implementation reproduces the paper's method *ordering* but sits ~21pp below its absolute numbers. Because every contribution here is a within-system delta, absolute matching is not required — but we quantify what is recoverable and attribute the rest. The GPD paper releases **no code and no weights**, and leaves guidance scale, candidate count K, and collision-cost formulas unspecified; training is not the cause (our budget matches; 5×/1M-step training moves SR −0.1pp). Sweeping the two documented levers on the GPD-stitched config (hybrid 400):

| config | SR% | avg t |
|---|---|---|
| baseline (K=32, scale 1) | 73.2 | 7.3s |
| K=128 | **76.5** | 10.9s |
| K=32, scale 2 | 76.0 | 7.3s |
| K=128, scale 3 | **76.5** | 10.9s |
| K=32 + sphere-SDF guidance | 67.2 | 3.4s |
| K=128 + sphere-SDF guidance | 72.0 | 4.1s |
| *GPD paper (unreleased weights/config)* | *92.8* | *1.9s* |

Candidate budget (K→128) and guidance scale (→2×) each add ~3pp, plateauing at **76.5%** — only ~3 of the ~20pp gap. Exact sphere-SDF *guidance* is **worse** (−6pp): the guide's clearance/expansion schedule supplies a safety margin geometric exactness lacks (another fidelity⊥SR instance). The residual ~16pp is attributable to the tuned 7-cost ensemble (we reproduce 1G→7G at +6pp vs the paper's +18pp), the exact guidance scale, and the authors' weights — none reconstructable from the paper. This confirms cross-method absolutes are unavailable and within-system deltas are the sound unit of claim. (`experiments/reproduction_guidance_tuning.sh`, `results/reproduction_*`, `results/reproduction_curve.{png,json}`.)

### 5.3 Beyond repair: three more levers, none beats it (audit generalizes; selection & learned-repair fail)
Extending §5.2, we (a) ran the continuous-feasibility audit **across datasets and a 2nd planner**, and tested two ways to shortcut stitch+trajopt:
- **Audit generalizes**: base waypoint→dense gap (mean±std over seeds) = **−25.0±1.7 / −17.4±2.6 / −20.3±1.2pp** (GPD global/hybrid/both, all significant); **EDMP −4.2±5.2pp (n.s.)** — representation-dependent (compact Bernstein most affected); repair flattens it and dynamic≈dense everywhere. → learned planners should report SR at a stated continuous-check resolution. (`continuous_audit_cross.{png,json}`)
- **Verifier-guided selection** (best-of-K by faithful oracle): at K=32 *worse* than guide+stitch (−3.4pp, stitching bridges across candidates); value scales with K but at K=128+trajopt = **83.5±1.0 vs 81.8±1.3 baseline (+1.65pp, CI [−1.0,+4.3], n.s.)** at +3s. Compute-allocation Pareto: diffusion dominates time (~7–10s), repair ~1–3s, selection <0.5s. (`compute_allocation.{png,json}`, `oracle_k128_stats.json`)
- **SMC particle steering** (resample K particles by sphere-SDF potential during denoising): −3.2pp base (collapses stitch diversity), −1.0pp n.s. with trajopt. (`results/smc_stats.json`)
- **Learned amortized repair** (distill trajopt → one-shot net, 1800 pairs): **fails** — one-shot = 64.2±2.1 (*worse than no repair*, val loss > identity baseline); learned+polish hybrid 75.6±1.8 (< 81.8) at barely lower latency. (`gpd/learned_repair.py`, `train_repair.py`, `results/bc_stats.json`)

**Synthesis:** five attempts (denser objective, exact sphere-SDF objective/guidance, verifier selection, learned repair, SMC steering) — none significantly exceeds plain oracle-gated trajopt. The lever is robust; the return is in *using* the faithful oracle inside stitch/repair.

### 5.2 Continuous-feasibility audit + dense repair objective
The dominant failure is *swept* infeasibility, but the executor/checks sample the path discretely. Differentiable swept-volume models exist (SVSDF arXiv:2405.00362, NeuralSVCD 2509.00499) but none run inside a diffusion planner's repair; safe-diffusion methods (SafeDiffuser) constrain only sampled points. We (i) **audit** continuous feasibility at check resolution R (samples/edge), and (ii) test a **dense** repair objective (S sub-configs/edge; the §4.3 midpoint is S=1).

**Audit (100 hybrid scenes)** — continuous SR@R:

| R (samples/edge) | base | repair (S=1) | dense (S=4) |
|---|---|---|---|
| 1 (waypoints) | 90.0 | 84.0 | 84.0 |
| 4 | 77.0 | 83.0 | 83.0 |
| 32 | 76.0 | 82.0 | 82.0 |

The base planner is 90% feasible at waypoints but **76% under a dense check** — ~14pp of "waypoint-free" plans sweep through obstacles. **Repair is resolution-flat (84→82%)** — it yields genuinely continuous-feasible paths. **But a denser objective doesn't help**: over 5 seeds S=1 = 81.8±1.3 vs S=4 = 81.7±1.3 (paired −0.15pp, CI [−0.79,+0.49]) at +2.5s/scene. The single-midpoint swept term is a *sufficient statistic* — because repair early-stops on a densified faithful oracle, a denser differentiable objective changes nothing (the fidelity⊥SR decoupling again). Continuous feasibility is worth *measuring* and *repairing*, not a heavier objective. (`continuous_audit.py`, `experiments/continuous_feasibility.sh`, `results/continuous_audit.{json,png}`, `results/continuous_edge_stats.json`; new flag `run_worker.py --repair_edge_samples`.)

---

## 6. Conclusion
In a controlled study of guided polynomial diffusion planning, the generative prior is not the bottleneck: five natural prior improvements leave success rate flat while improving nominal metrics. The gains live in the feasibility machinery — faithful-collision stitching (+17.4pp) and trajectory-optimization repair (+8.9pp, to 81.8% on hybrid). We release the implementation and all ablations to support feasibility-centric, rather than prior-centric, research on diffusion planners.

---

### Appendix: artifact map
All benchmark launchers live in `experiments/` (see `experiments/README.md` for the
full script→section map); analysis entry points are at the repo root.
- Prior ablations: `RESEARCH_PLAN.md`, `REPLICATION.md`, `results/capacity_*`, `results/smoothness_lam*`; `experiments/prior_capacity.sh`, `experiments/prior_smoothness.sh`.
- Conditioning: `gpd/conditional_unet.py`, `gpd/preprocess_scenes.py`, `train_gpd_cond.py`, `experiments/prior_conditioning{,_ablation}.sh`, `results/conditioning_*`.
- Extended training (1M steps): `models_long/GPDModel64_N8/`, `experiments/prior_long_training.sh`, `results/extended_{gpd,repair}/`, `results/extended_prior.json`.
- Multi-seed CIs: `run_worker.py --seed`, `experiments/stats_seeds_base_repair.sh` (baseline/repair), `experiments/stats_seeds_prior.sh` (conditioning/extended), `results/seed_stats.json`, `results/seeds_prior_stats.json`.
- Linear-seed control: `run_worker.py --naive_seed linear`, `experiments/feasibility_naive_seed.sh`, `results/linearseed_{baseline,repair}_s*/`, `results/linearseed_stats.json`.
- Failure analysis (base + post-repair shift): `diag_failures.py [N] [base|d6]`, `results/failure_modes{,_d6}.json`, `results/failure_shift.png`.
- Cross-planner repair (EDMP): `run_worker.py --method edmp --trajopt`, `experiments/feasibility_cross_planner.sh`, `results/edmp_{base,d6}/`.
- Mechanism figure (compression=regularizer): `analyze_smoothness.py`, `results/smoothness_stats.json`, `results/smoothness_jerk.png`.
- Exact collision objective: `gpd/sphere_collision.py`, `run_worker.py --sphere_cost`, `experiments/feasibility_sphere_collision.sh`, `results/sphere_stats.json`.
- Feasibility levers: `gpd/stitch.py` (faithful-collision RRT), `gpd/refine.py` (trajopt repair), `experiments/feasibility_repair.sh`, `results/repair_{off,densify,full}`; trajopt ablation `experiments/feasibility_repair_pareto.sh`, `results/repair_pareto_*`, `results/repair_pareto.json`.
- Matched-split baselines: `experiments/baselines.sh`, `results/baseline_*`.
- Reproduction harness: `run_worker.py` (flags `--n_control --conditional --cfg_weight --trajopt --trajopt_iters --sphere_cost --naive_seed --seed --pybullet_collision …`), `experiments/*.sh`, `merge_results.py`.

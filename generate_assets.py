"""
Generate all figures for BLOG.md and README.md.
Output: assets/*.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json, os

os.makedirs('assets', exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
EDMP_C  = '#2563EB'   # blue
GPD_C   = '#16A34A'   # green
BG      = '#F8FAFC'
GRID_C  = '#E2E8F0'
TEXT_C  = '#1E293B'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG,
    'axes.edgecolor':   GRID_C,
    'axes.labelcolor':  TEXT_C,
    'xtick.color':      TEXT_C,
    'ytick.color':      TEXT_C,
    'grid.color':       GRID_C,
    'text.color':       TEXT_C,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ── Data ───────────────────────────────────────────────────────────────────────
scene_types  = ['Tabletop', 'Cubby', 'Merged\nCubby', 'Dresser', 'Overall']
edmp_sr      = [95.2, 60.0, 48.3, 48.2, 65.8]
gpd_sr       = [96.2, 40.7, 34.7, 43.3, 59.1]
edmp_time    = [7.42, 6.77, 6.55, 10.71, 8.27]
gpd_time     = [1.89, 1.72, 1.68,  2.48, 2.02]
n_scenes     = [600, 300, 300, 600, 1800]


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Success Rate Comparison (bar chart)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(BG)

x      = np.arange(len(scene_types))
width  = 0.35
bars_e = ax.bar(x - width/2, edmp_sr, width, label='EDMP',
                color=EDMP_C, alpha=0.88, zorder=3)
bars_g = ax.bar(x + width/2, gpd_sr,  width, label='GPD',
                color=GPD_C,  alpha=0.88, zorder=3)

# Separate the "Overall" group visually
ax.axvline(3.5, color=GRID_C, lw=1.5, ls='--', zorder=2)

# Value labels
for bar in bars_e:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f'{h:.1f}%',
            ha='center', va='bottom', fontsize=9, color=EDMP_C, fontweight='bold')
for bar in bars_g:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f'{h:.1f}%',
            ha='center', va='bottom', fontsize=9, color=GPD_C, fontweight='bold')

# Scene counts as x-axis secondary labels
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(x)
ax2.set_xticklabels([f'n={n}' for n in n_scenes], fontsize=8, color='#64748B')
ax2.tick_params(top=False)
ax2.spines['top'].set_visible(False)

ax.set_xticks(x)
ax.set_xticklabels(scene_types, fontsize=11)
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_ylim(0, 108)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
ax.set_title('EDMP vs GPD — Success Rate by Scene Type\n(1800 scenes, RTX 4090)',
             fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('assets/success_rate_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved assets/success_rate_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Planning Time Comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)

bars_e = ax.bar(x - width/2, edmp_time, width, label='EDMP',
                color=EDMP_C, alpha=0.88, zorder=3)
bars_g = ax.bar(x + width/2, gpd_time,  width, label='GPD',
                color=GPD_C,  alpha=0.88, zorder=3)

ax.axvline(3.5, color=GRID_C, lw=1.5, ls='--', zorder=2)

for bar in bars_e:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f'{h:.2f}s',
            ha='center', va='bottom', fontsize=9, color=EDMP_C, fontweight='bold')
for bar in bars_g:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f'{h:.2f}s',
            ha='center', va='bottom', fontsize=9, color=GPD_C, fontweight='bold')

# Speedup annotation on Overall bar
speedup = edmp_time[-1] / gpd_time[-1]
ax.annotate(f'{speedup:.1f}× faster',
            xy=(x[-1] + width/2, gpd_time[-1]),
            xytext=(x[-1] + width/2 + 0.45, gpd_time[-1] + 2.5),
            fontsize=10, color=GPD_C, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GPD_C, lw=1.5))

ax.set_xticks(x)
ax.set_xticklabels(scene_types, fontsize=11)
ax.set_ylabel('Avg Planning Time (s/scene)', fontsize=12)
ax.set_ylim(0, 14)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=11, framealpha=0.9, loc='upper left')
ax.set_title('EDMP vs GPD — Average Planning Time by Scene Type\n(1800 scenes, RTX 4090)',
             fontsize=13, fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig('assets/planning_time_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved assets/planning_time_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Combined dashboard (SR + time side by side)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5.5))
fig.patch.set_facecolor(BG)
gs  = GridSpec(1, 2, figure=fig, wspace=0.32)

# Left: success rate
ax1 = fig.add_subplot(gs[0])
b1e = ax1.bar(x - width/2, edmp_sr, width, label='EDMP', color=EDMP_C, alpha=0.88, zorder=3)
b1g = ax1.bar(x + width/2, gpd_sr,  width, label='GPD',  color=GPD_C,  alpha=0.88, zorder=3)
ax1.axvline(3.5, color=GRID_C, lw=1.2, ls='--', zorder=2)
for bar, val in zip(list(b1e)+list(b1g),
                    [f'{v:.1f}%' for v in edmp_sr] + [f'{v:.1f}%' for v in gpd_sr]):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.9,
             val, ha='center', va='bottom', fontsize=7.5,
             color=EDMP_C if bar in b1e else GPD_C, fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(scene_types, fontsize=10)
ax1.set_ylabel('Success Rate (%)', fontsize=11)
ax1.set_ylim(0, 112); ax1.yaxis.grid(True, zorder=0); ax1.set_axisbelow(True)
ax1.legend(fontsize=10, framealpha=0.9)
ax1.set_title('Success Rate', fontsize=12, fontweight='bold')

# Right: planning time
ax2 = fig.add_subplot(gs[1])
b2e = ax2.bar(x - width/2, edmp_time, width, label='EDMP', color=EDMP_C, alpha=0.88, zorder=3)
b2g = ax2.bar(x + width/2, gpd_time,  width, label='GPD',  color=GPD_C,  alpha=0.88, zorder=3)
ax2.axvline(3.5, color=GRID_C, lw=1.2, ls='--', zorder=2)
for bar, val in zip(list(b2e)+list(b2g),
                    [f'{v:.2f}s' for v in edmp_time] + [f'{v:.2f}s' for v in gpd_time]):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.18,
             val, ha='center', va='bottom', fontsize=7.5,
             color=EDMP_C if bar in b2e else GPD_C, fontweight='bold')
ax2.annotate('4.1× faster', xy=(x[-1]+width/2, gpd_time[-1]),
             xytext=(x[-1]+width/2+0.4, gpd_time[-1]+2.8),
             fontsize=9.5, color=GPD_C, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=GPD_C, lw=1.4))
ax2.set_xticks(x); ax2.set_xticklabels(scene_types, fontsize=10)
ax2.set_ylabel('Avg Planning Time (s/scene)', fontsize=11)
ax2.set_ylim(0, 15); ax2.yaxis.grid(True, zorder=0); ax2.set_axisbelow(True)
ax2.legend(fontsize=10, framealpha=0.9, loc='upper left')
ax2.set_title('Planning Time', fontsize=12, fontweight='bold')

fig.suptitle('EDMP vs GPD — Full 1800-Scene Benchmark (RTX 4090, April 2026)',
             fontsize=13, fontweight='bold', y=1.02)
plt.savefig('assets/benchmark_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved assets/benchmark_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Bernstein basis illustration
# ══════════════════════════════════════════════════════════════════════════════
from math import comb

def bernstein_basis(n, t_vals):
    """Returns (len(t_vals), n) basis matrix."""
    T = len(t_vals)
    B = np.zeros((T, n))
    for k in range(n):
        B[:, k] = comb(n-1, k) * (t_vals**k) * ((1-t_vals)**(n-1-k))
    return B

t = np.linspace(0, 1, 300)
n = 8
B = bernstein_basis(n, t)

# Create a sample 1-DOF trajectory in joint space
np.random.seed(42)
control_pts = np.array([-1.2, -0.8, -0.3, 0.4, 0.9, 0.5, 0.1, 0.6])
waypoints   = B @ control_pts   # (300,)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

# Left: basis functions
ax = axes[0]
colors_b = plt.cm.tab10(np.linspace(0, 0.9, n))
for k in range(n):
    ax.plot(t, B[:, k], color=colors_b[k], lw=2.0, label=f'$B_{{{k}}}$')
ax.axvline(0, color='#94A3B8', lw=1, ls='--')
ax.axvline(1, color='#94A3B8', lw=1, ls='--')
ax.set_xlabel('Normalised trajectory time', fontsize=11)
ax.set_ylabel('Basis value', fontsize=11)
ax.set_title('Bernstein Basis Functions (M=8)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc='upper center')
ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
ax.set_facecolor(BG)

# Right: control points → trajectory
ax = axes[1]
ctrl_t = np.linspace(0, 1, n)
ax.plot(t, waypoints, color=EDMP_C, lw=2.5, label='Expanded trajectory (50 pts)', zorder=3)
ax.plot(ctrl_t, control_pts, 'o--', color=GPD_C, lw=1.5, ms=9,
        label='Control points (8 pts)', zorder=4)
# Highlight start and goal
ax.plot([0], [control_pts[0]], 's', color='#DC2626', ms=12, zorder=5, label='Start')
ax.plot([1], [control_pts[-1]], '*', color='#7C3AED', ms=14, zorder=5, label='Goal')
ax.set_xlabel('Normalised trajectory time', fontsize=11)
ax.set_ylabel('Joint angle (rad)', fontsize=11)
ax.set_title('8 Control Points → 50 Waypoints\nvia q = α @ B.T', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
ax.set_facecolor(BG)

fig.suptitle('Bernstein Polynomial Parameterisation in GPD',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('assets/bernstein_basis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved assets/bernstein_basis.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Diffusion chain comparison (EDMP T=255 vs GPD T=64)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 4))
fig.patch.set_facecolor(BG)

# Simulate a variance schedule (cosine)
def cosine_alpha_bar(T):
    steps = np.arange(T+1)
    return np.cos(((steps/T) + 0.008) / 1.008 * np.pi/2)**2

T_edmp, T_gpd = 255, 64
ab_edmp = cosine_alpha_bar(T_edmp)
ab_gpd  = cosine_alpha_bar(T_gpd)

t_edmp_norm = np.linspace(0, 1, T_edmp+1)
t_gpd_norm  = np.linspace(0, 1, T_gpd+1)

ax.plot(t_edmp_norm, ab_edmp, color=EDMP_C, lw=2.5,
        label=f'EDMP  T=255, dim=50×7=350')
ax.plot(t_gpd_norm,  ab_gpd,  color=GPD_C,  lw=2.5, ls='--',
        label=f'GPD   T=64,  dim=8×7=56')

# Shade the "extra" steps EDMP has
ax.fill_between(t_edmp_norm, ab_edmp, alpha=0.08, color=EDMP_C)
ax.fill_between(t_gpd_norm,  ab_gpd,  alpha=0.08, color=GPD_C)

ax.set_xlabel('Normalised denoising progress (t/T)', fontsize=11)
ax.set_ylabel('Signal power  ᾱ(t)', fontsize=11)
ax.set_title('Cosine Variance Schedule: EDMP (T=255) vs GPD (T=64)\n'
             'GPD needs 4× fewer steps over a 6× smaller latent space  →  ~25× less compute',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
ax.set_facecolor(BG)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('assets/diffusion_schedule.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved assets/diffusion_schedule.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — GPU utilisation / pipeline diagram (simple text-art as raster)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 3.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12); ax.set_ylim(0, 4)
ax.axis('off')

def box(x, y, w, h, color, label, sub='', ax=ax):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.08', facecolor=color, edgecolor='white',
        linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2 + (0.15 if sub else 0), label,
            ha='center', va='center', fontsize=9.5,
            fontweight='bold', color='white', zorder=4)
    if sub:
        ax.text(x+w/2, y+h/2-0.25, sub,
                ha='center', va='center', fontsize=7.5, color='#E2E8F0', zorder=4)

# Row 1: Original EDMP pipeline
ax.text(0.1, 3.3, 'Original EDMP (per step):', fontsize=9, color='#64748B',
        va='center', style='italic')
box(1.5, 2.8, 1.6, 0.8, '#6B7280', 'Model fwd', 'GPU', ax)
box(3.3, 2.8, 1.6, 0.8, '#DC2626', 'X_t→CPU', '510×/scene', ax)
box(5.1, 2.8, 1.6, 0.8, '#6B7280', 'grad (CPU)', 'numpy', ax)
box(6.9, 2.8, 1.6, 0.8, '#DC2626', 'CPU→GPU', '510×/scene', ax)
box(8.7, 2.8, 1.6, 0.8, '#6B7280', 'Update', 'GPU', ax)
for xpos in [3.1, 4.9, 6.7, 8.5]:
    ax.annotate('', xy=(xpos+0.2, 3.2), xytext=(xpos, 3.2),
                arrowprops=dict(arrowstyle='->', color='#94A3B8', lw=1.3))
ax.text(11.0, 3.2, '~22s/scene', ha='center', va='center',
        fontsize=9, color='#DC2626', fontweight='bold')

# Row 2: GPU-native pipeline
ax.text(0.1, 1.8, 'GPU-native (this work):', fontsize=9, color='#64748B',
        va='center', style='italic')
box(1.5, 1.3, 1.6, 0.8, '#2563EB', 'Model fwd', 'GPU', ax)
box(3.3, 1.3, 1.6, 0.8, '#16A34A', 'p_sample', 'GPU tensor ops', ax)
box(5.1, 1.3, 2.4, 0.8, '#7C3AED', 'get_gradient', '1 CPU call/step', ax)
box(7.7, 1.3, 1.6, 0.8, '#16A34A', 'Update α_t', 'GPU', ax)
for xpos in [3.1, 4.9, 7.5]:
    ax.annotate('', xy=(xpos+0.2, 1.7), xytext=(xpos, 1.7),
                arrowprops=dict(arrowstyle='->', color='#94A3B8', lw=1.3))
ax.text(11.0, 1.7, '~2s/scene\n(GPD)', ha='center', va='center',
        fontsize=9, color='#16A34A', fontweight='bold')

ax.set_title('Denoising Pipeline: CPU Round-trips Eliminated',
             fontsize=12, fontweight='bold', pad=8)

plt.tight_layout()
plt.savefig('assets/pipeline_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved assets/pipeline_diagram.png")

print("\nAll assets generated:")
for f in sorted(os.listdir('assets')):
    size = os.path.getsize(f'assets/{f}') // 1024
    print(f"  assets/{f}  ({size} KB)")

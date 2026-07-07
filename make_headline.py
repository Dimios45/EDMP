"""Regenerate the headline delta figure (paper Fig.1) from the CI'd results."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# (label, delta, ci_halfwidth or None, group)  group: 'prior' (red) / 'feas' (blue)
rows = [
    ("5$\\times$ training",   -0.10, 1.61, 'prior'),
    ("noise schedule",        -3.0,  None, 'prior'),
    ("conditioning (CFG)",    -1.75, 0.62, 'prior'),
    ("capacity ($n{=}16$)",   -6.3,  None, 'prior'),
    ("EDMP $+$ repair",       +7.6,  2.35, 'feas'),
    ("trajopt repair",        +8.9,  1.30, 'feas'),
    ("faithful stitch",      +17.4,  2.17, 'feas'),
]
rows = sorted(rows, key=lambda r: r[1])
labels=[r[0] for r in rows]; vals=[r[1] for r in rows]
err=[r[2] if r[2] is not None else 0 for r in rows]
colors=['#c0504d' if r[3]=='prior' else '#4472c4' for r in rows]
fig,ax=plt.subplots(figsize=(6.6,3.5))
y=list(range(len(rows)))
ax.barh(y, vals, xerr=err, color=colors, alpha=0.9,
        error_kw=dict(ecolor='0.3', capsize=3, lw=1))
ax.axvline(0, color='k', lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('$\\Delta$ success rate (pp) vs. baseline')
for i,r in enumerate(rows):
    e = r[2] if r[2] is not None else 0
    off = e + 0.7
    x = r[1] + (off if r[1] >= 0 else -off)
    tag = f"{r[1]:+.1f}" + ('' if r[2] is not None else '$^{*}$')
    ax.text(x, i, tag, va='center', ha='left' if r[1] >= 0 else 'right', fontsize=8)
ax.legend(handles=[Patch(color='#c0504d',label='prior-side'),
                   Patch(color='#4472c4',label='feasibility machinery')],
          fontsize=8, loc='lower right')
ax.set_xlim(-11, 23); ax.grid(axis='x', alpha=0.3)
ax.set_title('$^{*}$single-run (no CI); all others 95% CI over seeds', fontsize=7.5, loc='left')
plt.tight_layout(); plt.savefig('paper/figs/headline_deltas.png', dpi=160); plt.close()
print("saved")

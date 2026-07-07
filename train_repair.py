"""
Train the learned amortized repair operator on (seed -> trajopt-repaired) pairs emitted
by `run_worker.py --save_repair_pairs`. Regresses the trajopt output from the stitched
seed + obstacle set. Held-out split for honest eval.

Usage: python train_repair.py [--epochs 4000] [--pairs_glob 'results/repair_pairs*/*.jsonl']
Saves models/LearnedRepair/weights.pt + losses.
"""
import os, glob, json, argparse, numpy as np, torch
import torch.nn as nn
from gpd.learned_repair import LearnedRepair


def load_pairs(pattern):
    seeds, reps, obss, masks, succ = [], [], [], [], []
    for fn in sorted(glob.glob(pattern)):
        for line in open(fn):
            p = json.loads(line)
            seeds.append(p['seed']); reps.append(p['repaired'])
            obss.append(p['scene_obs']); masks.append(p['scene_mask'])
            succ.append(int(p.get('success', 1)))
    return (np.array(seeds, np.float32), np.array(reps, np.float32),
            np.array(obss, np.float32), np.array(masks, bool), np.array(succ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--pairs_glob', default='results/repair_pairs*/*.jsonl')
    ap.add_argument('--val_frac', type=float, default=0.15)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    seeds, reps, obss, masks, succ = load_pairs(args.pairs_glob)
    n = len(seeds); assert n > 0, f"no pairs matched {args.pairs_glob}"
    print(f"loaded {n} pairs | success rate of trajopt targets: {succ.mean():.2f}")
    rng = np.random.default_rng(0); idx = rng.permutation(n)
    nval = int(n * args.val_frac); vi, ti = idx[:nval], idx[nval:]
    to = lambda a, i, dt=torch.float32: torch.tensor(a[i], dtype=dt, device=dev)
    S, R = to(seeds, ti), to(reps, ti)
    O, M = to(obss, ti), to(masks, ti, torch.bool)
    Sv, Rv = to(seeds, vi), to(reps, vi)
    Ov, Mv = to(obss, vi), to(masks, vi, torch.bool)

    model = LearnedRepair().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    nt = len(ti); losses = []
    for e in range(args.epochs):
        model.train()
        b = torch.randint(0, nt, (min(args.batch, nt),), device=dev)
        pred = model(S[b], O[b], M[b])
        loss = lossf(pred, R[b])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if e % 500 == 0 or e == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                vl = lossf(model(Sv, Ov, Mv), Rv).item()
                # identity baseline (predict seed): how much does the seed already match?
                idl = lossf(Sv, Rv).item()
            print(f"epoch {e:5d}  train {loss.item():.5f}  val {vl:.5f}  (identity val {idl:.5f})", flush=True)
    model.save()
    np.save(os.path.join(model.model_dir, 'losses.npy'), np.array(losses))
    print("saved", model.model_dir)


if __name__ == '__main__':
    main()

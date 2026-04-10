"""
Merge per-worker JSON result files into a single summary.

Usage:
    python merge_results.py --method gpd --num_workers 4 \
                            --results_dir results --out compare_results.json
"""

import argparse, json, glob
import numpy as np

SCENE_TYPES = ['tabletop', 'cubby', 'merged_cubby', 'dresser']


def merge(method, results_dir, num_workers):
    all_results = []
    for wid in range(num_workers):
        path = f'{results_dir}/{method}_w{wid}.json'
        try:
            with open(path) as f:
                data = json.load(f)
            all_results.extend(data['results'])
        except FileNotFoundError:
            print(f"  WARNING: {path} not found — skipping worker {wid}")

    # Sort by scene_type order then scene_num
    order = {st: i for i, st in enumerate(SCENE_TYPES)}
    all_results.sort(key=lambda r: (order.get(r['scene_type'], 99), r['scene_num']))
    return all_results


def print_summary(results, method):
    if not results:
        print(f"  {method}: no results")
        return
    sr  = sum(r['success'] for r in results)
    tot = len(results)
    avg = np.mean([r['plan_time'] for r in results])
    print(f"\n{'='*60}")
    print(f"{method.upper()} SUMMARY")
    print(f"{'='*60}")
    print(f"  Total: {sr}/{tot}  ({100*sr/tot:.1f}%)  avg {avg:.2f}s/scene")
    print(f"\n  Per scene-type:")
    for st in SCENE_TYPES:
        sub = [r for r in results if r['scene_type'] == st]
        if sub:
            s = sum(r['success'] for r in sub)
            a = np.mean([r['plan_time'] for r in sub])
            print(f"    {st:15s}  {s}/{len(sub)} ({100*s/len(sub):.1f}%)  avg {a:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',      choices=['edmp', 'gpd', 'both'], default='both')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--out',         default='compare_results.json')
    args = parser.parse_args()

    combined = {}
    methods = ['gpd', 'edmp'] if args.method == 'both' else [args.method]

    for m in methods:
        r = merge(m, args.results_dir, args.num_workers)
        combined[m] = r
        print_summary(r, m)

    if 'edmp' in combined and 'gpd' in combined and combined['edmp'] and combined['gpd']:
        avg_e = np.mean([r['plan_time'] for r in combined['edmp']])
        avg_g = np.mean([r['plan_time'] for r in combined['gpd']])
        print(f"\n  Speedup: {avg_e/avg_g:.1f}× faster with GPD")

    with open(args.out, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved → {args.out}")


if __name__ == '__main__':
    main()

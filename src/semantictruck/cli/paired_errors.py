"""Test whether the predicted segmentation's error is equivalent to the reference segmentation's.

    semantictruck-paired-errors --table results/pipeline/results_per_truck.csv
    semantictruck-paired-errors --margin 0.5 --alpha 0.05

Pairs each truckload's absolute leave-one-out percentage error under the two arms and reads the
difference (predicted minus reference) three ways: a Hodges-Lehmann shift with its exact Wilcoxon
confidence interval, a BCa bootstrap interval for the mean as a sensitivity check, and a TOST
equivalence test against a margin of +/- `--margin` percentage points.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from semantictruck.arms import PREDICTED, REFERENCE

DEFAULT_TABLE = 'results/pipeline/results_per_truck.csv'
ALPHA = 0.05
DELTA = 1.0


def signed_rank_null_cdf(n: int) -> np.ndarray:
    """Exact null distribution of the Wilcoxon signed-rank statistic T+, assuming no ties.

    Returns the cumulative probabilities P(T+ <= t) for t = 0 .. n(n+1)/2.
    """
    max_t = n * (n + 1) // 2
    counts = np.zeros(max_t + 1, dtype=np.float64)
    counts[0] = 1.0
    for k in range(1, n + 1):
        counts[k:] = counts[k:] + counts[:-k].copy()
    return np.cumsum(counts) / 2.0 ** n


def hodges_lehmann_ci(d: np.ndarray, conf: float = 0.95) -> tuple[float, float, float, float]:
    """Hodges-Lehmann shift of `d` and its exact confidence interval from the Walsh averages.

    Returns (estimate, lower, upper, achieved coverage); the coverage is the exact one, which the
    discreteness of T+ puts at or above `conf`.
    """
    d = np.asarray(d, float)
    n = d.size
    i, j = np.triu_indices(n)                  # all pairs i <= j
    walsh = np.sort((d[i] + d[j]) / 2.0)       # the n(n+1)/2 Walsh averages
    m = walsh.size
    hl = np.median(walsh)

    a = 1 - conf
    cdf = signed_rank_null_cdf(n)
    # Largest k with P(T+ <= k) <= a/2, giving CI = [W_(k+1), W_(M-k)] in 1-based order statistics
    k = int(np.searchsorted(cdf, a / 2, side='right')) - 1
    lo, hi = walsh[k], walsh[m - k - 1]
    achieved = 1 - 2 * cdf[k]
    return hl, lo, hi, achieved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='semantictruck-paired-errors', description=__doc__.split('\n\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--table', default=DEFAULT_TABLE,
                        help='results_per_truck.csv from a finished run with both arms')
    parser.add_argument('--alpha', type=float, default=ALPHA,
                        help='significance level of each one-sided TOST test')
    parser.add_argument('--margin', type=float, default=DELTA,
                        help='equivalence margin, in percentage points of absolute error')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    alpha, delta = args.alpha, args.margin

    if not os.path.exists(args.table):
        print('no per-truck table at {} -- run semantictruck-pipeline first'.format(args.table))
        return 1

    # Absolute percentage error per truckload and arm, paired on load_id so a load missing from
    # one arm drops out of both rather than shifting every pair after it
    table = pd.read_csv(args.table)
    ape = table.pivot(index='load_id', columns='segmentation', values='error_loo_pct').abs()
    if REFERENCE not in ape.columns or PREDICTED not in ape.columns:
        print('{} does not carry both arms ({}, {}); there is nothing to pair'.format(
            args.table, REFERENCE, PREDICTED))
        return 1
    ape = ape[[REFERENCE, PREDICTED]].dropna()

    d = (ape[PREDICTED] - ape[REFERENCE]).to_numpy()
    n = d.size

    print('{} paired truckload(s)'.format(n))
    print('Mean diff = {:.4f}, median diff = {:.4f}'.format(d.mean(), np.median(d)))
    print('Zeros in d: {}, tied |d|: {}'.format((d == 0).sum(), n - np.unique(np.abs(d)).size))

    hl, lo95, hi95, cov95 = hodges_lehmann_ci(d, 0.95)
    print('\n--- Hodges-Lehmann (matches Wilcoxon signed-rank) ---')
    print('HL estimate of paired difference = {:.4f}'.format(hl))
    print('95% CI = [{:.4f}, {:.4f}]  (exact coverage {:.4f})'.format(lo95, hi95, cov95))

    # Bootstrap CI for the mean difference, as a sensitivity check on the rank-based interval
    boot = stats.bootstrap((d,), np.mean, confidence_level=0.95, n_resamples=10_000,
                           method='BCa', random_state=42)
    print('\n--- Bootstrap (BCa) for the mean difference ---')
    print('Mean diff = {:.4f}, 95% CI = [{:.4f}, {:.4f}]'.format(
        d.mean(), boot.confidence_interval.low, boot.confidence_interval.high))

    # TOST with the Wilcoxon signed-rank test:
    #   H0a: shift <= -delta  vs  H1a: shift > -delta
    #   H0b: shift >= +delta  vs  H1b: shift < +delta
    p_lower = stats.wilcoxon(d + delta, alternative='greater').pvalue
    p_upper = stats.wilcoxon(d - delta, alternative='less').pvalue
    p_tost = max(p_lower, p_upper)

    # The equivalent CI formulation: the (1 - 2 alpha) HL interval lies inside (-delta, +delta)
    _, lo90, hi90, _ = hodges_lehmann_ci(d, 1 - 2 * alpha)
    inside = lo90 > -delta and hi90 < delta

    print('\n--- TOST equivalence, margin +/-{} ---'.format(delta))
    print('p (lower bound) = {:.4g}'.format(p_lower))
    print('p (upper bound) = {:.4g}'.format(p_upper))
    print('TOST p = {:.4g}  ->  {}'.format(
        p_tost, 'EQUIVALENT' if p_tost < alpha else 'equivalence NOT shown'))
    print('{:.0%} HL CI = [{:.4f}, {:.4f}] {} (-{}, {})'.format(
        1 - 2 * alpha, lo90, hi90, 'inside' if inside else 'NOT inside', delta, delta))

    # The smallest margin equivalence would be declared at -- worth reporting whatever the verdict
    print('Smallest margin supported at alpha={}: +/-{:.4f}'.format(
        alpha, max(abs(lo90), abs(hi90))))
    return 0


if __name__ == '__main__':
    sys.exit(main())

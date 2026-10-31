"""
Q2: do regional subreddits scale with real population?

Joins per-subreddit activity against census populations, fits a power law
via log-log linear regression, and ranks subreddits by residual (who's
over/under-performing relative to their city's population).

Usage:
    python analyze_population.py <activity-csv> <regional-csv> <results-dir>
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def main(activity_csv, regional_csv, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    activity = pd.read_csv(activity_csv)
    regional = pd.read_csv(regional_csv)

    merged = regional.merge(activity, on='subreddit', how='inner')
    merged = merged[(merged['population'] > 0) & (merged['total_activity'] > 0)].copy()

    log_pop = np.log10(merged['population'])
    log_activity = np.log10(merged['total_activity'])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_pop, log_activity)

    merged['predicted_log_activity'] = intercept + slope * log_pop
    merged['residual'] = log_activity - merged['predicted_log_activity']

    ranking = merged[['subreddit', 'city', 'population', 'total_activity', 'residual']] \
        .sort_values('residual', ascending=False)
    ranking.to_csv(f'{results_dir}/residual_ranking.csv', index=False)

    with open(f'{results_dir}/population_fit_summary.txt', 'w') as f:
        f.write(f'power law: activity ~ population^{slope:.4f}\n')
        f.write(f'slope (exponent): {slope:.4f}\n')
        f.write(f'intercept: {intercept:.4f}\n')
        f.write(f'r_squared: {r_value ** 2:.4f}\n')
        f.write(f'p_value: {p_value:.4e}\n')
        f.write(f'std_err: {std_err:.4f}\n')
        f.write(f'n subreddits: {len(merged)}\n')

    print(f'Power law exponent: {slope:.3f} (r^2={r_value ** 2:.3f}, p={p_value:.2e}, n={len(merged)})')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(log_pop, log_activity)
    for _, row in merged.iterrows():
        ax.annotate(row['subreddit'], (np.log10(row['population']), np.log10(row['total_activity'])), fontsize=7)
    xs = np.linspace(log_pop.min(), log_pop.max(), 100)
    ax.plot(xs, intercept + slope * xs, color='red', label=f'fit: slope={slope:.2f}, $R^2$={r_value ** 2:.2f}')
    ax.set_xlabel('log10(CMA population)')
    ax.set_ylabel('log10(reddit activity)')
    ax.set_title('Regional subreddit activity vs. population')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{results_dir}/population_powerlaw.png')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

"""
Q1: what makes a submission successful?

"Success" = landing in the top quartile of score *within its subreddit*,
since raw score is heavily right-skewed and scales with subreddit size.
Stickied posts are dropped first since mod-pinned posts get inflated scores
for reasons unrelated to content.

Beyond the global model, we also stratify by subreddit category (news,
memes, advice, technical — see data/subreddit_categories.csv) and ask
whether the *same* features predict success everywhere, or whether e.g. a
question-mark title helps in advice subreddits but is irrelevant in meme
subreddits. Category coverage is a small, manually-curated set of large
subreddits, not the whole corpus — that's a limitation, not a sample of
"all subreddits by category".

Usage:
    python analyze_success.py <submissions-parquet-dir> <results-dir> [categories-csv]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

FEATURES = ['hour', 'weekday', 'title_length', 'has_question_mark', 'all_caps_ratio', 'is_self']
MIN_GROUP_SIZE = 200  # below this, per-category stats/model results are too noisy to report
MIN_SUBREDDIT_POSTS = 50  # below this, "top quartile within subreddit" isn't a meaningful label


def all_caps_ratio(title):
    words = [w for w in title.split() if w.isalpha()]
    if not words:
        return 0.0
    caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    return caps / len(words)


def filter_small_subreddits(df, min_posts):
    post_counts = df['subreddit'].value_counts()
    kept = post_counts[post_counts >= min_posts].index
    return df[df['subreddit'].isin(kept)]


def build_features(df):
    dt = pd.to_datetime(df['created_utc'], unit='s', utc=True)
    df['hour'] = dt.dt.hour
    df['weekday'] = dt.dt.weekday
    df['is_weekend'] = df['weekday'] >= 5
    df['title_length'] = df['title'].str.len()
    df['has_question_mark'] = df['title'].str.contains(r'\?', regex=True).astype(int)
    df['all_caps_ratio'] = df['title'].apply(all_caps_ratio)
    df['is_self'] = df['is_self'].astype(int)
    return df


def run_stats_tests(df, label):
    groups = [g['score_pctile'].values for _, g in df.groupby('hour')]
    kw_stat, kw_p = stats.kruskal(*groups)

    weekend = df.loc[df['is_weekend'], 'score_pctile']
    weekday = df.loc[~df['is_weekend'], 'score_pctile']
    mw_stat, mw_p = stats.mannwhitneyu(weekend, weekday, alternative='two-sided')

    return {
        'label': label,
        'n': len(df),
        'kw_stat': kw_stat, 'kw_p': kw_p,
        'mw_stat': mw_stat, 'mw_p': mw_p,
        'weekend_median': weekend.median(), 'weekday_median': weekday.median(),
    }


def format_stats_result(result):
    return (
        f'--- {result["label"]} (n={result["n"]}) ---\n'
        f'Kruskal-Wallis score_pctile by hour: H={result["kw_stat"]:.4f}, p={result["kw_p"]:.6e}\n'
        f'Mann-Whitney score_pctile weekend vs weekday: U={result["mw_stat"]:.4f}, p={result["mw_p"]:.6e}\n'
        f'weekend median score_pctile: {result["weekend_median"]:.4f}\n'
        f'weekday median score_pctile: {result["weekday_median"]:.4f}\n'
    )


def train_classifier(df, label):
    X = df[FEATURES]
    y = df['success']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=353, stratify=y
    )

    clf = GradientBoostingClassifier(random_state=353)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    importances = pd.Series(clf.feature_importances_, index=FEATURES, name=label)
    return report, importances


def main(submissions_dir, results_dir, categories_csv='data/subreddit_categories.csv'):
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_parquet(submissions_dir)
    df = df[df['stickied'] == False].copy()
    df = df.dropna(subset=['title', 'score', 'created_utc', 'subreddit'])

    n_before, subs_before = len(df), df['subreddit'].nunique()
    df = filter_small_subreddits(df, MIN_SUBREDDIT_POSTS).copy()
    dropped_rows = n_before - len(df)
    print(
        f'Dropping {subs_before - df["subreddit"].nunique()} subreddits with '
        f'< {MIN_SUBREDDIT_POSTS} posts ({dropped_rows} rows, {dropped_rows / n_before:.1%} of data)'
    )

    df = build_features(df)

    # Normalize score within subreddit so cross-subreddit comparisons (and
    # the stats tests below) aren't dominated by a handful of huge subs.
    df['score_pctile'] = df.groupby('subreddit')['score'].rank(pct=True)
    df['success'] = (df['score_pctile'] >= 0.75).astype(int)

    # --- Global analysis (all subreddits pooled) ---
    overall_stats = run_stats_tests(df, 'overall')
    print(f'Kruskal-Wallis (score by hour): H={overall_stats["kw_stat"]:.2f}, p={overall_stats["kw_p"]:.4e}')
    print(f'Mann-Whitney (weekend vs weekday): U={overall_stats["mw_stat"]:.2f}, p={overall_stats["mw_p"]:.4e}')

    overall_report, overall_importances = train_classifier(df, 'overall')
    print(overall_report)

    with open(f'{results_dir}/stats_summary.txt', 'w') as f:
        f.write(format_stats_result(overall_stats))
    with open(f'{results_dir}/classifier_report.txt', 'w') as f:
        f.write(overall_report)
    overall_importances.to_csv(f'{results_dir}/feature_importances.csv')

    fig, ax = plt.subplots(figsize=(7, 5))
    overall_importances.sort_values().plot.barh(ax=ax)
    ax.set_xlabel('Feature importance')
    ax.set_title('What predicts a top-quartile submission?')
    fig.tight_layout()
    fig.savefig(f'{results_dir}/feature_importances.png')

    hourly = df.groupby('hour')['score_pctile'].mean()
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    hourly.plot(ax=ax2, marker='o')
    ax2.set_xlabel('Posting hour (UTC)')
    ax2.set_ylabel('Mean within-subreddit score percentile')
    ax2.set_title('Score percentile by posting hour')
    fig2.tight_layout()
    fig2.savefig(f'{results_dir}/score_by_hour.png')

    # --- Stratified analysis: does the same feature set matter equally
    # across subreddit categories, or is e.g. a question-mark title only
    # helpful in advice subreddits? ---
    categories = pd.read_csv(categories_csv)
    categories['subreddit_lower'] = categories['subreddit'].str.lower()
    df['subreddit_lower'] = df['subreddit'].str.lower()

    df_cat = df.merge(categories[['subreddit_lower', 'category']], on='subreddit_lower', how='inner')

    stats_rows = []
    importances_by_category = {}
    skipped = []

    for category, group in df_cat.groupby('category'):
        if len(group) < MIN_GROUP_SIZE:
            skipped.append((category, len(group)))
            continue
        stats_rows.append(run_stats_tests(group, category))
        _, importances = train_classifier(group, category)
        importances_by_category[category] = importances

    with open(f'{results_dir}/stats_by_category.txt', 'w') as f:
        for row in stats_rows:
            f.write(format_stats_result(row) + '\n')
        for category, n in skipped:
            f.write(f'--- {category}: skipped, only {n} rows (< {MIN_GROUP_SIZE}) ---\n')

    if importances_by_category:
        importance_table = pd.DataFrame(importances_by_category)
        importance_table.to_csv(f'{results_dir}/feature_importances_by_category.csv')

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        im = ax3.imshow(importance_table.values, cmap='viridis', aspect='auto')
        ax3.set_xticks(range(len(importance_table.columns)))
        ax3.set_xticklabels(importance_table.columns, rotation=30, ha='right')
        ax3.set_yticks(range(len(importance_table.index)))
        ax3.set_yticklabels(importance_table.index)
        for i in range(len(importance_table.index)):
            for j in range(len(importance_table.columns)):
                ax3.text(j, i, f'{importance_table.values[i, j]:.2f}', ha='center', va='center',
                         color='white' if importance_table.values[i, j] < importance_table.values.max() * 0.6 else 'black',
                         fontsize=8)
        fig3.colorbar(im, ax=ax3, label='Feature importance')
        ax3.set_title('Feature importance by subreddit category')
        fig3.tight_layout()
        fig3.savefig(f'{results_dir}/feature_importances_by_category.png')

    if skipped:
        print(f'Skipped categories with too little data: {skipped}')


if __name__ == '__main__':
    categories_arg = sys.argv[3] if len(sys.argv) > 3 else 'data/subreddit_categories.csv'
    main(sys.argv[1], sys.argv[2], categories_arg)

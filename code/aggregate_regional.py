"""
Build a per-subreddit activity table from the extracted corpus, for the Q2
population-scaling analysis.

Usage:
    python aggregate_regional.py <extract-dir> <output-csv>

<extract-dir> is the local copy of the Spark output (contains
submissions/ and comments/ parquet subdirectories).

Bot and deleted/removed authors are dropped before counting unique
participants, since they'd otherwise inflate activity for subreddits with
heavy automod/bot traffic.
"""
import sys

import pandas as pd

EXCLUDED_AUTHORS = {'[deleted]', '[removed]', 'automoderator'}


def excluded_mask(authors):
    # Vectorized equivalent of "author is missing/blank, in EXCLUDED_AUTHORS,
    # or ends in 'bot'" — avoids a Python-level .apply() over every row,
    # which is the slowest possible way to do this on a full-corpus scale.
    lname = authors.fillna('').astype(str).str.lower()
    return (lname == '') | lname.isin(EXCLUDED_AUTHORS) | lname.str.endswith('bot')


def main(extract_dir, output_csv):
    submissions = pd.read_parquet(
        f'{extract_dir}/submissions', columns=['subreddit', 'author', 'score', 'num_comments']
    )
    comments = pd.read_parquet(f'{extract_dir}/comments', columns=['subreddit', 'author', 'score'])

    submissions = submissions[~excluded_mask(submissions['author'])]
    comments = comments[~excluded_mask(comments['author'])]

    sub_agg = submissions.groupby('subreddit').agg(
        num_submissions=('author', 'size'),
        submission_score_sum=('score', 'sum'),
    )
    comment_agg = comments.groupby('subreddit').agg(
        num_comments=('author', 'size'),
    )

    activity = sub_agg.join(comment_agg, how='outer').fillna(0)

    all_authors = pd.concat([
        submissions[['subreddit', 'author']],
        comments[['subreddit', 'author']],
    ])
    activity['unique_authors'] = all_authors.groupby('subreddit')['author'].nunique()
    activity['unique_authors'] = activity['unique_authors'].fillna(0)

    activity['total_activity'] = activity['num_submissions'] + activity['num_comments']
    activity = activity.reset_index()

    activity.to_csv(output_csv, index=False)
    print(f'Wrote {len(activity)} subreddits to {output_csv}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

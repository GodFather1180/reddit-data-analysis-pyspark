# Reddit Analysis: what makes a post succeed, and do city subreddits scale with population?

A two question analysis of the SFU course cluster's Reddit corpus (submissions and comments, 2022 and 2023), built around a PySpark extraction job and a local pandas and scikit-learn analysis.

## Status

| Part | State |
|---|---|
| PySpark extraction job (`code/spark_extract.py`) | Done. Ran on the course YARN cluster. |
| Cleaning, feature engineering, statistical tests, classifier, power law fit (`code/`) | Written and unit tested |
| Unit tests (`tests/`) | 9 pass. `test_analyze_population.py` specifies `fit_power_law`, which is not implemented yet. |
| Results from the full 2022 and 2023 extract | **Not generated yet.** The extract is on the cluster, not in this repo, so there are no figures or metrics to show. |

Everything that depends on the full run (row counts, test statistics, model scores, the fitted exponent) is deliberately left out rather than guessed. The commands to produce it are below.

## The two questions

**Q1. What makes a submission successful?** Raw score is heavily right skewed and depends on how big a subreddit is, so a post counts as successful if it lands in the **top quartile of score within its own subreddit**. Features: posting hour, weekday, title length, a question mark in the title, share of all caps words, and whether it is a text post.
- Kruskal-Wallis test of score percentile across posting hours
- Mann-Whitney U test of weekend against weekday posts
- Gradient boosted classifier with feature importances, run on everything and again by subreddit category (news and discussion, memes and images, advice and support, technical and professional) to see whether the same features matter everywhere. The category map is a small hand picked set of 16 large subreddits, so it is a subset and not a sample of all subreddits.

**Q2. Do regional subreddits scale with population?** Join per subreddit activity (submissions plus comments, bots removed) to 2021 Census populations for 24 Canadian metropolitan areas, fit `activity = c * population^b` as a log-log regression, report the exponent `b`, and rank cities by residual to see who is more or less active than their size predicts. This is the part that combines the Reddit corpus with an unrelated source (Statistics Canada).

## Data

- Reddit corpus: Hive partitioned by year and month in HDFS at `/courses/datasets/reddit_submissions_repartitioned/` and `/courses/datasets/reddit_comments_repartitioned/`. `created_utc` is a string in this corpus, so the job casts it.
- [data/regional_subreddits.csv](data/regional_subreddits.csv): 24 city subreddits mapped to 2021 CMA populations. I checked the figures against Statistics Canada table 98-10-0005-01 and corrected one (Guelph is 165,588).
- [data/subreddit_categories.csv](data/subreddit_categories.csv): the category map for the stratified Q1 analysis.

The raw corpus and the extract are not committed.

## The extraction job and what made it fast

[code/spark_extract.py](code/spark_extract.py) keeps 10 submission columns and 5 comment columns and writes snappy Parquet, so the whole thing fits on a laptop. Getting it to run well on a small cluster (4 executors with 1 core and 1 GB each) took three changes:

1. **Explicit schema.** The first version called `spark.read.json` on the corpus root and let Spark infer types, which meant sampling records across every year from 2006 to 2025. I stopped that run after more than 13 minutes without it getting past the read. Giving Spark a schema for just the fields I need skips inference entirely.
2. **Narrow read path.** Instead of reading the root and filtering on `year`, the job globs straight to `year={2022,2023}`.
3. **`basePath` option.** A multi year glob failed with a conflicting directory structures error until `basePath` was set to the corpus root, which keeps `year` as a partition column.

After those changes a one year smoke test (2006) finished in under a minute. A full run over 2022 and 2023 completed on the cluster in about 37 minutes (from the YARN application logs). One thing worth knowing: YARN reported 10% progress for that entire run, so progress percentage is not a reliable signal for Spark jobs. The `_SUCCESS` marker in the output directory is.

## Cleaning decisions

- Drop `[deleted]`, `[removed]`, `AutoModerator`, and any author ending in `bot` before counting unique participants. Bot and moderator traffic is large and uneven across communities.
- Drop stickied posts from the success model, since moderators pin them and they collect inflated scores for reasons unrelated to content.
- Drop rows missing a title, score, timestamp or subreddit.
- Subreddits with fewer than 200 posts after filtering are skipped in the stratified analysis instead of reporting unstable estimates.

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ --ignore=tests/test_analyze_population.py
```

On the cluster:

```bash
spark-submit code/spark_extract.py reddit-extract
hdfs dfs -copyToLocal reddit-extract .
```

Locally, from this folder:

```bash
python code/aggregate_regional.py reddit-extract data/activity.csv
python code/analyze_population.py data/activity.csv data/regional_subreddits.csv results/
python code/analyze_success.py reddit-extract/submissions results/
```

Outputs land in `results/`: `residual_ranking.csv`, `population_powerlaw.png`, `population_fit_summary.txt`, `stats_summary.txt`, `classifier_report.txt`, `feature_importances.csv` and `.png`, `score_by_hour.png`, plus the per category files.

## Known limitations

- The classifier report gives precision, recall, F1 and accuracy. It does not compute AUC.
- A city subreddit is not the same thing as a census metropolitan area, so some residuals measure how well the mapping fits, not only how active a community is.
- The population join matches subreddit names exactly and case sensitively. Reddit keeps the original capitalisation, so check that all 24 rows matched before trusting the fit.
- With 24 cities, one unusual city can move the exponent noticeably.

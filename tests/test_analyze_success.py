import pandas as pd

from analyze_success import all_caps_ratio, build_features, filter_small_subreddits


def test_filter_small_subreddits_drops_subreddits_below_threshold():
    df = pd.DataFrame({
        'subreddit': ['big'] * 5 + ['small'] * 2,
        'score': range(7),
    })

    result = filter_small_subreddits(df, min_posts=3)

    assert set(result['subreddit'].unique()) == {'big'}
    assert len(result) == 5


def test_filter_small_subreddits_keeps_subreddits_at_exactly_the_threshold():
    df = pd.DataFrame({
        'subreddit': ['exact'] * 3,
        'score': range(3),
    })

    result = filter_small_subreddits(df, min_posts=3)

    assert len(result) == 3


def test_all_caps_ratio_counts_fully_uppercase_words_only():
    assert all_caps_ratio('BREAKING news today') == 1 / 3


def test_all_caps_ratio_ignores_single_letter_words():
    # a lone "I" or "A" shouldn't count as shouting
    assert all_caps_ratio('I am fine') == 0.0


def test_all_caps_ratio_empty_title_is_zero():
    assert all_caps_ratio('') == 0.0


def test_build_features_extracts_hour_weekday_and_weekend_flag():
    df = pd.DataFrame({
        # 2024-01-06 is a Saturday, 08:00 UTC
        'created_utc': [1704528000],
        'title': ['Is this real?'],
        'is_self': [True],
    })

    result = build_features(df)

    assert result.loc[0, 'hour'] == 8
    assert result.loc[0, 'weekday'] == 5  # Saturday
    assert bool(result.loc[0, 'is_weekend']) is True
    assert result.loc[0, 'has_question_mark'] == 1
    assert result.loc[0, 'title_length'] == len('Is this real?')

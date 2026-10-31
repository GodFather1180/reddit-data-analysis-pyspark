import pandas as pd

from aggregate_regional import excluded_mask


def test_excluded_mask_flags_known_bad_authors():
    authors = pd.Series(['[deleted]', '[removed]', 'AutoModerator', 'real_user'])

    assert list(excluded_mask(authors)) == [True, True, True, False]


def test_excluded_mask_flags_bot_suffixed_authors_case_insensitively():
    authors = pd.Series(['NewsBot', 'weatherbot', 'robotics_fan'])

    assert list(excluded_mask(authors)) == [True, True, False]


def test_excluded_mask_flags_missing_authors():
    authors = pd.Series([None, '', 'real_user'])

    assert list(excluded_mask(authors)) == [True, True, False]

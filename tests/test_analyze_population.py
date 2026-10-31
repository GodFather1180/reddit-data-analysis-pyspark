import numpy as np
import pandas as pd
import pytest

from analyze_population import fit_power_law


def test_fit_power_law_recovers_known_exponent():
    population = np.array([1_000, 10_000, 100_000, 1_000_000, 10_000_000], dtype=float)
    true_slope = 0.5
    activity = 10 * population ** true_slope
    df = pd.DataFrame({'population': population, 'total_activity': activity})

    result_df, stats = fit_power_law(df)

    assert stats['slope'] == pytest.approx(true_slope, abs=1e-6)
    assert stats['r_squared'] == pytest.approx(1.0, abs=1e-6)
    assert (result_df['residual'].abs() < 1e-6).all()


def test_fit_power_law_ranks_overperformers_first():
    df = pd.DataFrame({
        'population': [1000, 1000, 1000],
        'total_activity': [100, 1000, 10],  # same population, different activity
    })

    result_df, _ = fit_power_law(df)

    # the row with the most activity relative to its (identical) population
    # should have the highest residual
    assert result_df.sort_values('residual', ascending=False)['total_activity'].iloc[0] == 1000

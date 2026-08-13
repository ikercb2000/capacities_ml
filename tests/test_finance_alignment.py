import pandas as pd
import pytest

from capacities_ml_fin.finance import (
    apply_publication_lag,
    point_in_time_join,
    validate_no_lookahead,
)


def test_publication_lag_creates_information_availability_dates():
    data = pd.DataFrame({"period_end": ["2024-03-31", "2024-06-30"]})

    result = apply_publication_lag(data, lag=5)

    expected = pd.to_datetime(["2024-04-05", "2024-07-05"])
    pd.testing.assert_series_equal(
        result["available_date"],
        pd.Series(expected, name="available_date"),
    )


def test_point_in_time_join_uses_only_available_information():
    left = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-05", "2024-01-10"])}
    )
    right = pd.DataFrame(
        {
            "available_date": pd.to_datetime(["2024-01-03", "2024-01-08"]),
            "value": [1.0, 2.0],
        }
    )

    result = point_in_time_join(left, right)

    assert result["value"].tolist() == [1.0, 2.0]
    assert validate_no_lookahead(result)


def test_lookahead_validation_reports_future_information():
    invalid = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05"]),
            "available_date": pd.to_datetime(["2024-01-06"]),
        }
    )

    assert not validate_no_lookahead(invalid, raise_on_error=False)
    with pytest.raises(ValueError, match="Look-ahead detected"):
        validate_no_lookahead(invalid)

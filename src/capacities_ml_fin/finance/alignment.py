# imports
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


# publication timing
def apply_publication_lag(
    data: pd.DataFrame,
    *,
    period_end: str = "period_end",
    lag: int | str | pd.Timedelta,
    available_col: str = "available_date",
) -> pd.DataFrame:
    """Create an availability date from a reporting-period date and fixed lag.

    This is a fallback for data without exact publication timestamps. A fixed
    lag does not reproduce true point-in-time data, so callers should prefer
    actual release dates whenever available.
    """
    if period_end not in data.columns:
        raise KeyError(f"Unknown period_end column: {period_end!r}.")

    if isinstance(lag, (int, np.integer)):
        if int(lag) < 0:
            raise ValueError("lag must be non-negative.")
        offset = pd.Timedelta(days=int(lag))
    else:
        offset = pd.to_timedelta(lag)
        if offset < pd.Timedelta(0):
            raise ValueError("lag must be non-negative.")

    result = data.copy()
    result[period_end] = pd.to_datetime(result[period_end], errors="raise")
    result[available_col] = result[period_end] + offset
    return result


# look-ahead validation
def validate_no_lookahead(
    data: pd.DataFrame,
    *,
    model_time: str = "date",
    available_time: str = "available_date",
    raise_on_error: bool = True,
) -> bool:
    """Check that no observation is used before its information became available."""
    for column in (model_time, available_time):
        if column not in data.columns:
            raise KeyError(f"Unknown time column: {column!r}.")

    model_dates = pd.to_datetime(data[model_time], errors="raise")
    available_dates = pd.to_datetime(data[available_time], errors="raise")
    valid_rows = model_dates.notna() & available_dates.notna()
    violations = valid_rows & (available_dates > model_dates)

    if violations.any() and raise_on_error:
        first = data.index[violations][0]
        raise ValueError(
            "Look-ahead detected: information is available after the model date "
            f"at row {first!r}."
        )
    return not bool(violations.any())


# point-in-time alignment
def point_in_time_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str = "date",
    available_on: str = "available_date",
    by: str | Sequence[str] | None = None,
    suffixes: tuple[str, str] = ("", "_right"),
    allow_exact_matches: bool = True,
) -> pd.DataFrame:
    """Join each model row to the most recent information available by that time.

    The operation is always backward-looking: a row dated ``t`` can only match
    a right-hand observation whose ``available_on`` timestamp is less than or
    equal to ``t``. Optional grouping keys make the operation suitable for
    asset-level panels such as fundamentals joined by ticker.
    """
    if left_on not in left.columns:
        raise KeyError(f"Unknown left time column: {left_on!r}.")
    if available_on not in right.columns:
        raise KeyError(f"Unknown availability column: {available_on!r}.")

    by_columns: list[str]
    if by is None:
        by_columns = []
    elif isinstance(by, str):
        by_columns = [by]
    else:
        by_columns = list(by)
        if not by_columns:
            raise ValueError("by must contain at least one grouping column.")

    for column in by_columns:
        if column not in left.columns or column not in right.columns:
            raise KeyError(f"Grouping column {column!r} must exist in both frames.")

    left_work = left.copy()
    right_work = right.copy()
    left_work[left_on] = pd.to_datetime(left_work[left_on], errors="raise")
    right_work[available_on] = pd.to_datetime(right_work[available_on], errors="raise")
    left_work["__pit_order__"] = np.arange(len(left_work))

    if not by_columns:
        joined = pd.merge_asof(
            left_work.sort_values(left_on),
            right_work.sort_values(available_on),
            left_on=left_on,
            right_on=available_on,
            direction="backward",
            allow_exact_matches=allow_exact_matches,
            suffixes=suffixes,
        )
    else:
        pieces: list[pd.DataFrame] = []
        grouper = by_columns[0] if len(by_columns) == 1 else by_columns
        right_groups = {
            key: group.copy()
            for key, group in right_work.groupby(grouper, sort=False, dropna=False)
        }

        for key, left_group in left_work.groupby(grouper, sort=False, dropna=False):
            right_group = right_groups.get(key)
            if right_group is None:
                template = right_work.iloc[:0].copy()
                right_group = template

            # Grouping columns are already known from the left frame. Remove them
            # from the right frame to avoid duplicate columns after the as-of join.
            right_payload = right_group.drop(columns=by_columns, errors="ignore")
            piece = pd.merge_asof(
                left_group.sort_values(left_on),
                right_payload.sort_values(available_on),
                left_on=left_on,
                right_on=available_on,
                direction="backward",
                allow_exact_matches=allow_exact_matches,
                suffixes=suffixes,
            )
            pieces.append(piece)

        joined = pd.concat(pieces, axis=0, ignore_index=True)

    joined = joined.sort_values("__pit_order__").drop(columns="__pit_order__")
    joined.index = left.index

    if available_on in joined.columns:
        validate_no_lookahead(
            joined,
            model_time=left_on,
            available_time=available_on,
        )
    return joined

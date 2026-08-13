# Point-in-time alignment

Point-in-time alignment controls **when information becomes usable by the model**. A reporting period end is not necessarily the date on which the information was public.

## Why availability dates matter

Suppose a quarterly accounting variable refers to period end 31 March but is published on 25 April. A model dated 10 April must not use that value.

The relevant condition is

\[
\text{available\_date}\le\text{model\_date}.
\]

The alignment utilities are built around this condition.

## `apply_publication_lag`

When exact release timestamps are unavailable, a fixed lag can create an approximate availability date:

```python
from capacities_ml_fin.finance import apply_publication_lag

fundamentals = apply_publication_lag(
    fundamentals,
    period_end="period_end",
    lag=45,
    available_col="available_date",
)
```

An integer lag is interpreted as calendar days. Strings and pandas Timedelta values are also accepted.

!!! warning
    A fixed publication lag is only a fallback. It does not reproduce true point-in-time data and can be materially wrong when actual filing delays vary across companies or periods.

## `point_in_time_join`

```python
from capacities_ml_fin.finance import point_in_time_join

model_frame = point_in_time_join(
    daily_market,
    fundamentals,
    left_on="date",
    available_on="available_date",
    by="ticker",
)
```

For each left-hand row at model date $t$, the function selects the most recent right-hand row whose `available_date` is at or before $t$.

The operation is a backward as-of join. With grouping keys such as ticker, each group is aligned independently.

### Exact matches

`allow_exact_matches=True` permits information with `available_date == date` to be used on that date. Set it to `False` if your empirical timing convention requires information to have been available strictly before the model timestamp.

## `validate_no_lookahead`

```python
from capacities_ml_fin.finance import validate_no_lookahead

validate_no_lookahead(
    model_frame,
    model_time="date",
    available_time="available_date",
)
```

If any valid row satisfies

\[
\text{available\_date}>\text{date},
\]

the default behavior is to raise a `ValueError` identifying the first violating row.

Use `raise_on_error=False` to receive a Boolean instead:

```python
is_valid = validate_no_lookahead(
    model_frame,
    raise_on_error=False,
)
```

## Example with multiple assets

```python
market = pd.DataFrame(
    {
        "date": [...],
        "ticker": [...],
        "return": [...],
    }
)

fundamentals = pd.DataFrame(
    {
        "available_date": [...],
        "ticker": [...],
        "profit_margin": [...],
    }
)

aligned = point_in_time_join(
    market,
    fundamentals,
    by="ticker",
)
```

The original left-row order and index are restored after the grouped as-of operation.

## A recommended empirical rule

Keep at least three separate concepts in your dataset:

- **period date** — what period a fundamental number describes;
- **availability date** — when the number became observable;
- **model date** — when the prediction/portfolio decision is formed.

Do not overwrite one with another. `point_in_time_join` is designed around this separation.

## API

See [Finance API — alignment](../api/finance.md#point-in-time-alignment).

# Grouped Histograms with `Table.hist`

This project supports grouped histograms via the `group` parameter on `Table.hist`. Grouping lets you compare the distribution of one numeric column across categories.

Minimal example:

```python
from datascience import Table, make_array

t = Table().with_columns(
    'height', make_array(160, 170, 180, 175),
    'gender', make_array('F', 'M', 'M', 'F')
)

# Compare height distributions by gender (overlaid)
t.hist('height', group='gender')

# Show the grouped histograms side by side
t.hist('height', group='gender', side_by_side=True)
```

Interpretation:
- When `group='gender'`, the table splits rows by each unique value in `gender` and draws a separate histogram for the `height` values in each group.
- Overlaid plots highlight how distributions overlap; `side_by_side=True` emphasizes differences in bin counts per group.

Notes and constraints:
- `group` cannot be used together with `bin_column`.
- `group` expects exactly one numeric value column (e.g., `'height'`). Passing multiple value columns raises a `ValueError`.
- If `group` does not reference an existing column label or index, a `ValueError` is raised.
- If the data are empty for all groups, `hist` creates an empty figure and returns without error.


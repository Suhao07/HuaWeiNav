# Active Observation

After arrival, call `verify_target`. The backend supplies an observation that satisfies the arrival barrier.

Use the verification result to choose the next query:

- `need_better_view`: Preserve the matched object pair, query alternative viewpoints for the same instance, and collect new evidence.
- `uncertain`: The available evidence is insufficient.
- `reject_candidate`: Evidence contradicts the candidate instance.
- `reject_relation`: The current object pair fails the required relation; this result applies only to that pair.

The domain reducer owns updates to instance counts.

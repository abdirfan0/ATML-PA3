# Recorded resources

| phase | updates | presentations | trainable_parameters | measured_seconds | peak_allocated_GiB | peak_reserved_GiB | timing_scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Feature caching | 0 | 6528 | 0 | 150.64900136299997 | 0.6581196784973145 |  | Feature-cache process session; staging, preprocessing, forward and persistence. Excludes smoke and main training. |
| Stage 1 | 2250 | 36000 | 1837056 | 1100.8001436249958 | 3.114654064178467 | 10.375 | Sum of update calls, including overflow retries; excludes development, checkpoint saves and model loading. |
| Stage 2 | 1500 | 24000 | 6424576 | 1178.7881800110022 | 3.093252658843994 | 10.44921875 | Sum of update calls, including overflow retries; excludes development, checkpoint saves and model loading. |
| Stage 3 | 750 | 12000 | 6424576 | 594.2121995770053 | 3.039240837097168 | 10.44921875 | Sum of update calls, including overflow retries; excludes development, checkpoint saves and model loading. |
| Final evaluation | 0 | 1280 | 0 | 71.06560166300005 | 1.8591012954711914 |  | Final evaluation session, including reference NLL, generation, bootstrap and output writes; excludes model loading. |

Total main-update time: 2873.80 s. Development evaluations: 39.33 s. Recovery-state writes: 6.65 s. Generation only (all 1280 outputs): 58.94 s. The logs do not record one complete queue wall-clock duration.

Feature forwards alone took 2.413 s; this is not the feature-caching wall time. Five initial AMP overflow attempts were retried on the same sampled batch before any successful update. Frozen weights and inference-only arrays are excluded from trainable counts.

# Residual-selected Fresh IID-100 × 3 ranking

All comparisons are paired by identical `data_seed`; the ranking metric is finite-only mean mesh-geodesic MAE, matching the prior project reports.

| Rank | Model | MAE geo, mm | MAE euc, mm | Non-finite geo |
|---:|---|---:|---:|---:|
| 1 | `trial81_honest_4965` | 4.3057 ± 0.0799 | 2.5864 ± 0.1383 | 17 |
| 2 | `trial81_push_3771` | 4.7815 ± 0.2607 | 2.6974 ± 0.2325 | 14 |
| 3 | `trial81_original` | 5.2502 ± 0.3628 | 2.9388 ± 0.1101 | 13 |

## Per-repeat paired results

- `trial81_honest_4965`, repeat 1, data_seed 202603260: geo 4.2527 mm; euc 2.4478 mm; non-finite geo 2.
- `trial81_honest_4965`, repeat 2, data_seed 202603261: geo 4.2456 mm; euc 2.5363 mm; non-finite geo 6.
- `trial81_honest_4965`, repeat 3, data_seed 202603262: geo 4.4187 mm; euc 2.7752 mm; non-finite geo 9.
- `trial81_original`, repeat 1, data_seed 202603260: geo 4.7376 mm; euc 2.8519 mm; non-finite geo 3.
- `trial81_original`, repeat 2, data_seed 202603261: geo 5.5250 mm; euc 3.0941 mm; non-finite geo 7.
- `trial81_original`, repeat 3, data_seed 202603262: geo 5.4880 mm; euc 2.8702 mm; non-finite geo 3.
- `trial81_push_3771`, repeat 1, data_seed 202603260: geo 4.5114 mm; euc 2.3801 mm; non-finite geo 2.
- `trial81_push_3771`, repeat 2, data_seed 202603261: geo 4.6990 mm; euc 2.9306 mm; non-finite geo 6.
- `trial81_push_3771`, repeat 3, data_seed 202603262: geo 5.1339 mm; euc 2.7815 mm; non-finite geo 6.

## Pairwise differences (A − B), geodesic mm

- `trial81_honest_4965` − `trial81_original`: [-0.4849, -1.2793, -1.0694], mean -0.9445; wins 3:0 (ties 0).
- `trial81_honest_4965` − `trial81_push_3771`: [-0.2587, -0.4534, -0.7153], mean -0.4758; wins 3:0 (ties 0).
- `trial81_original` − `trial81_push_3771`: [+0.2262, +0.8260, +0.3541], mean +0.4688; wins 0:3 (ties 0).

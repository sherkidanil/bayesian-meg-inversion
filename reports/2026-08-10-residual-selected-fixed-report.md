# Residual-selected dataset evaluation

- Dataset: `/mnt/local/tester/meg/new_version_mesh_experiments/best_examples/test_dataset_mesh.npz`
- Explicit localization G: `/mnt/local/tester/meg/reports/leadfield_provenance_2x2_20260810_final/rebuilt_leadfield_geometry.npz` key `G`
- Model: `trial81_push_3771`; samples: 300; draws: 100
- Selection uses sensor data, G, mesh, and dataset-level spatial parameters; truth is used only after selection for metrics.

- MAE euc: 2.9302 mm
- MAE geo: 5.4065 mm (16 non-finite)

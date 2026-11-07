# Repository candidate scope

The local publication candidate is limited to an explicit file whitelist in
`release_manifest.json`; repository visibility and publication require a
separate decision. The candidate contains:

- reusable estimator, OT-map, inference, metric, vectorization, and data APIs
  under `src/otrank/`;
- the frozen linear, ReLU, and NHANES configurations under `configs/`;
- the simulation and real-data runners;
- the empirical paper-output CLI and notebook.
- compact frozen metrics in `paper_data/`, with no participant-level records.

Development tests, archives, local audits/controllers, exploratory results,
private large artifacts, caches and credentials are excluded from the candidate.
The centered-IF migration is retained as an explicitly scoped maintenance tool,
not a new estimator or a finite-temperature Hessian correction.

Default output generation reads only the current `paper_data/manifest.json`
and its ten named CSVs, and writes these three domains:

1. `results/simulation/linear_n1000_p20_d10_final/`;
2. `results/simulation/relu_n1000_p8_d4_final/`;
3. `results/real_data/nhanes_p43_final/`.

Frozen metrics reproduce the paper tables and figures without model fitting.
Linear and ReLU runners expose separate estimator refits. The real-data
entrypoint defaults to aggregate/fold-level frozen outputs, with an explicit
full-retained-source maintenance option. No one-command full NHANES OTRR
refit workflow or public hosting of private fitted-state bundles is claimed.

Rotation replay stores seeds and generator/device/dtype/runtime metadata, not
new Q matrices. It refuses mismatched runtimes or unavailable CUDA rather than
regenerating a different matrix on CPU. Frozen paper outputs do not replay Q.

The predictor list is machine-readable; its grouped manuscript layout is not
part of this data-driven table generator. Output index keys are semantic and
do not depend on the manuscript's changing table/figure numbers.

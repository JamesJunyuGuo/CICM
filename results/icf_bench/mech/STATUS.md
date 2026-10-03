# Stage K K2/K3 Mechanism Status

Updated: 2026-07-21 00:10 America/Indianapolis

## Current State

Mechanism extraction/analysis code is prepared and smoke/full jobs are queued.
No Stage-J files and no K1d artifacts were modified.

## Jobs

- Smoke H100: `19424538`, script `results/icf_bench/mech/k2k3_smoke_h100.slurm`, output `results/icf_bench/mech/smoke/`
- Smoke A100 duplicate: `19424539`, script `results/icf_bench/mech/k2k3_smoke_a100.slurm`, output `results/icf_bench/mech/smoke_a100/`
- Full H100: `19424595`, script `results/icf_bench/mech/k2k3_full_h100.slurm`, output `results/icf_bench/mech/full_h100/`, dependency `afterok:19424538`
- Full A100 duplicate: `19424594`, script `results/icf_bench/mech/k2k3_full_a100.slurm`, output `results/icf_bench/mech/full_a100/`, dependency `afterok:19424538`

The full jobs will not run unless H100 smoke succeeds.

## Validation Already Done

- `bash -n` passed for all Slurm/finalize scripts.
- `python -m py_compile src/icf_mech.py src/icf_mech_extract.py src/icf_mech_label.py src/icf_mech_analyze.py` passed.
- `pytest tests/test_icf_mech.py tests/test_icf_k1d.py tests/test_icf_matchers.py -q` passed: 21 tests.
- CPU prompt span sanity passed for DP old/new preference spans and IF forget/value spans on first examples.

## After Full Harvest Completes

Run from the repo root on the login node with `OPENROUTER_API_KEY` set:

```bash
bash results/icf_bench/mech/k2k3_finalize.sh
```

That script chooses the first complete full harvest directory, runs DP local hybrid judge labels, then writes:

- `results/icf_bench/mech/k2k3_summary.json`
- `results/icf_bench/mech/REPORT.md`


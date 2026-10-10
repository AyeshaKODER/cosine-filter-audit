# Cosine Filter Audit

This repository contains a Flower/PyTorch simulation for auditing hard-threshold
cosine filtering. It implements two explicitly separate representations:

- **`delta`** (the primary experiment): scores the floating-point client update
  `local_state - received_global_state`; an attack scales that displacement
  before constructing the submitted state.
- **`legacy_full_model`**: preserves the historical full-state scoring and
  scaling path for comparison. Do not pool its results with delta-space runs.

The project does not establish that this implementation is FedVault, CosDefense,
or another upstream defense. See
[`docs/code_to_manuscript_map.md`](docs/code_to_manuscript_map.md) for exact
semantics, evidence limits, and manuscript-alignment notes.

The code includes an experimental multi-signal update-scoring candidate,
seeded Dirichlet label-skew partitions, and two explicitly separate adaptive
attack modes for a follow-up study. The candidate does not use server training
data; the centralized test set is used only to report task performance. Neither
the candidate nor attacks are validated research results, and the configured
multi-signal cutoff is provisional.

## Requirements

- Python 3.11 or newer (required by Flower 1.36)
- Flower and the dependencies declared in `pyproject.toml`
- Internet access on the first run if CIFAR-10 is not already cached
- CPU is sufficient for the smoke run; a GPU can accelerate training

Create and activate an environment, then install the project from this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Model-stack dependency versions are pinned in `pyproject.toml`; Flower packages
are constrained to compatible release series. Each run record stores the exact
Python, Flower, PyTorch, and NumPy versions visible to the server process, and
records the Git commit and working-tree state when available.

## Validate without training

Run the offline unit suite; it does not download CIFAR-10 or start Flower:

```powershell
python -m unittest discover -s tests -v
```

## Smoke run

The following one-round run checks the end-to-end simulation, evaluation, and
run-artifact writing. It is a **plumbing check only**, not research evidence.
On a cache miss it downloads CIFAR-10.

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
flwr run . --stream --run-config "num-server-rounds=1 selection-mode='none' attack-start-round=999 save-model=false"
```

Each invocation writes `<run-output-dir>/<run_id>/run.json`. The default
`run-output-dir='runs'` is anchored to Flower's installed app directory; set an
absolute `run-output-dir` in `flwr run --run-config` to save artifacts in this
repository's `runs/` folder. The file records configuration, available
software/data provenance, initial-model hash, per-round client scores and
selections, aggregation weights, diagnostics, round status, and server
evaluation metrics. A missing Git revision or unpinned dataset revision is
recorded as missing rather than inferred. Treat the smoke result as a code-path
check only.

## Experiment semantics

In delta mode, the reference is the unweighted mean of valid submitted client
deltas (including the attacker) when `reference-mode="current_mean"`. A client
is flagged only when its score is **strictly below** `cosine-threshold`.
Oracle/benign-only controls use the server-designated Flower node ID in this
IID simulation; the strategy never trusts a client-supplied maliciousness
metric. The delta-mode attacker defaults to the lowest registered node ID and
the selected ID is recorded in the run artifact.

Delta-mode attacks alter floating-point tensors as
`global + attack_scale * (local - global)`. Non-floating buffers are copied
unchanged from the received global model. FedAvg aggregates selected submitted
states using their sample counts, renormalized over selected clients.

Selection modes are `none`, `detector`, `multi_signal`, `oracle`,
`random_same_size`, and `benign_only`. The multi-signal candidate combines
robustly standardized update distance from the current geometric median,
log-update norm, temporal drift, and client score-history deviation. Missing
history signals are omitted and the other available signals receive equal
weight. Its cutoff is `multi-signal-threshold`; tune it on development seeds
and freeze it before test runs. The current default of `3.5` is provisional,
not an empirically selected value.

Data partitioning is configured with `partitioning-mode='iid'` or
`partitioning-mode='dirichlet'`. The Dirichlet path partitions labels using
`dirichlet-alpha` and `partition-seed`. Available severities for the proposed
study are IID, α=1.0, α=0.3, and α=0.1; no training runs have been launched
for these settings. Delta references are `current_mean` and
`previous_accepted_delta`; legacy full-model references are `current_mean` and
`previous_global_model`. Historical-reference modes use the current mean on
round one.

Adaptive attacks are enabled with `adaptive-attack-mode='phase1_own_history'`
or `adaptive-attack-mode='phase2_cohort_aware'`, together with
`selection-mode='multi_signal'` and an explicit `attack-start-round`.
Phase 1 optimizes cross-entropy on the malicious client's own training data
and projects each candidate update against a surrogate score built from that
client's update history. Its score is only an approximation and is not
numerically interchangeable with the cohort detector score. Phase 2 adds an
extra server-to-attacker message with
the current peer updates, then projects against the exact current-cohort
multi-signal scoring function. Phase 2 is an intentionally strong
white-box simulation: it discloses peer updates to the attacker and should not
be described as a standard FL privacy-preserving protocol. Both modes log
attack scores, local loss change, optimization steps, and constraint status.
The attack is opt-in; the default mode is `none`.

## Paired comparison suites

The Flower server supports `run-suite=true`. All arms run sequentially against
the same live Flower Grid, with a fixed attacker node, seed-reset initial
models, partition-keyed client RNG streams, and the configured data split seed.
The suite writes one `run.json` per arm and a `suite_manifest.json` under
`<run-output-dir>/paired_suites/`. A reported node-to-partition mapping is
checked for stability and recorded; ground-truth attacker labels still derive
only from the designated server node ID.

Supported selection arms are `none`, `cosine`, `multi_signal`, `oracle`,
`random_same_size`, and `benign_only`. Update-space aggregation arms are
`krum`, `multi_krum`, `coordinate_median`, and `trimmed_mean`. The default
suite configuration spans three attack conditions, six selection arms, and
four robust baselines, so use narrow development configurations first.
Five seeds × four partition regimes × 30 arms × 10 rounds is a large training
job; do not launch a sweep until its cost is approved.

The following is a plumbing-only example, not enough data to calibrate or
freeze a cutoff:

```powershell
flwr run . --stream --run-config "run-suite=true suite-phase='development' suite-seeds='1' suite-regimes='iid' suite-attack-modes='none,phase1_own_history,phase2_cohort_aware' suite-selection-arms='multi_signal' suite-aggregation-arms='' num-server-rounds=1 run-output-dir='C:\path\to\cosine-filter-audit\runs'"
```

A cutoff-eligibility development suite requires five seeds, all four regimes,
no attack, and multi-signal selection (20 runs). Confirmatory seeds must be
disjoint from calibration seeds. This is a training workload; get approval
before launching it.

```powershell
flwr run . --stream --run-config "run-suite=true suite-phase='development' suite-seeds='1,2,3,4,5' suite-regimes='iid,dirichlet-1.0,dirichlet-0.3,dirichlet-0.1' suite-attack-modes='none' suite-selection-arms='multi_signal' suite-aggregation-arms='' num-server-rounds=3 run-output-dir='C:\path\to\cosine-filter-audit\runs'"
$manifest = Get-Content 'C:\path\to\cosine-filter-audit\runs\paired_suites\<suite-id>\suite_manifest.json' | ConvertFrom-Json
$records = $manifest.run_files | ForEach-Object { $_.path }
python -m cosine_filter_audit.calibration @records --max-benign-fpr 0.05 --output calibration.json
```

The cutoff is selected only from benign-only scores under the 5% false
rejection cap; adaptive scores do not influence threshold selection. A
calibration record becomes eligible after covering IID plus Dirichlet
α=1.0/0.3/0.1 and at least five development seeds. Confirmatory runs reject
records that are not marked
`ready_for_confirmatory`; set `suite-phase='confirmatory'`,
`suite-calibration-source` to that JSON path, and
`multi-signal-threshold` to its recorded cutoff. Use disjoint confirmatory
seeds. After freezing, adaptive attacks are configured against the frozen
threshold and evaluated on those separate seeds.

Summarize paired seed differences and bootstrap intervals from a suite:

```powershell
python -m cosine_filter_audit.analysis <suite_manifest.json> --output summary.json
```

This reports benign false rejection, attacker detection, accuracy/loss, round
statuses, and paired effect estimates. Development suites are for calibration;
confirmatory claims require an independently frozen threshold and enough
seeds. It has completed a one-seed, three-round, four-regime plumbing pilot;
the output is not confirmatory evidence. Further development or confirmatory
sweeps still run CIFAR-10 training and require explicit approval.

See [`docs/experiment_protocol.md`](docs/experiment_protocol.md) before
producing paper results. It defines the control arms, staged run plan, and
current limitations on strict pairing.

## Linux runs and resuming an interrupted suite

Use a Linux machine with a persistent project directory and persistent storage
for both `runs/` and the Flower/Ray cache. The project does not provision cloud
resources or automatically configure Ray GPU resources. A Linux move does not
make the suite cheaper by itself; the full 600-arm, ten-round confirmatory
design is a substantial training workload. Do not start it until the dataset,
storage, compute budget, frozen calibration, and runtime are approved.
Flower may also resolve the app's dependencies in its isolated run
environment; the first training run can download CIFAR-10 if it is not cached.

Create the environment and install the project on the Linux host (this may
download dependencies and CIFAR-10 when the first simulation starts):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
```

Start a small approved suite from the project root, writing output to durable
storage and preserving the full console log:

```bash
set -o pipefail
mkdir -p runs/logs
flwr run . --stream --run-config "run-suite=true suite-phase='development' suite-seeds='1' suite-regimes='iid' suite-attack-modes='none' suite-selection-arms='multi_signal' suite-aggregation-arms='' num-server-rounds=1 run-output-dir='runs'" 2>&1 | tee runs/logs/suite.log
```

Each completed arm is recorded in `suite_manifest.json` using an atomic
manifest replacement. After interruption, pass that manifest as
`suite-resume-manifest`; keep every scientific setting unchanged:

```bash
flwr run . --stream --run-config "run-suite=true suite-phase='development' suite-seeds='1' suite-regimes='iid' suite-attack-modes='none' suite-selection-arms='multi_signal' suite-aggregation-arms='' num-server-rounds=1 run-output-dir='runs' suite-resume-manifest='runs/paired_suites/<suite-id>/suite_manifest.json'"
```

Resume verifies the source fingerprint, scientific configuration, calibration
hash and threshold, runtime versions, partition IDs, and every completed run
artifact before skipping it. Resume in the same Python/Flower/PyTorch/NumPy
environment; changing code, package versions, seed/regime/arm lists, or other
scientific settings requires a new suite. Copy the entire suite directory if
moving to another host, and retain its relative run-artifact paths. Analysis
refuses an incomplete suite manifest; inspect individual run records for
diagnostics, but do not report a partial suite as complete.

## Historical artifacts

`results/legacy_full_model/` contains the timestamped JSON and CSV records from
the earlier implementation, along with its hard-coded plotting source. They
lack sufficient provenance for pairing or current delta-space claims. Do not
present those artifacts as delta-space results. The smoke and five-arm
exploratory pilot artifacts are under `results/experiment_artifacts/` with
their limitations documented; they are not confirmatory findings. The adjacent
`diagram/` folder is outside this repository; the ignore rule also excludes a
folder with that name if later copied under this root.

## Layout

```text
cosine_filter_audit/  Flower client/server apps, strategy, tensor utilities
frontend/             Local interactive research UI
tests/                Offline unit tests
docs/                 Experiment protocol and code/manuscript evidence map
results/              Legacy archive and curated exploratory run artifacts
runs/                 Per-run JSON/model artifacts (generated; ignored)
```

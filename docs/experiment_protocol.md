# Minimal experiment protocol

This protocol separates a code-path smoke test from evidence-producing
experiments. Archived JSON and CSV files under
`results/legacy_full_model/` are not paired controls for the new implementation.

## Proposed follow-up study (not yet validated)

The initial scope is centralized synchronous FedAvg on CIFAR-10, with no
trusted server training data used by the detector. The centralized CIFAR-10
test set is reserved for outcome reporting. Begin with untargeted poisoning;
targeted backdoors, asynchronous FL, and decentralized FL are out of scope.
Compare an IID control with seeded Dirichlet label-skew partitions at
α ∈ {1.0, 0.3, 0.1}.

The current `multi_signal` implementation is a **candidate heuristic**, not an
established robust aggregator. It computes a current-round geometric-median
reference and equally averages available positive robust-z anomaly signals:
distance from that reference, absolute log-norm deviation, temporal drift from
the client's prior update, and deviation from the client's historical score.
Temporal and history signals are unavailable during the first round. The
default cutoff (`3.5`) is only a placeholder. The calibration command selects
a cutoff using **benign-only** multi-signal client scores, choosing the least
restrictive threshold that meets the predeclared false-rejection cap (default
5%). Adaptive attack scores are excluded from calibration because Phase 2 is
explicitly optimized against the configured threshold; using those attack
scores to select and then evaluate the same boundary is circular. Calibration
eligibility requires IID and all three required Dirichlet regimes and at least
five development seeds. Confirmatory seeds must be disjoint from the
calibration seeds. A small one-seed plumbing run must not be used to freeze a
threshold. The detector itself receives no labels or server training data.

Both adaptive attack modes are now implemented. They maximize cross-entropy on
the attacker's own local training batches and project each candidate update
back into the configured score constraint. Phase 1 (`phase1_own_history`)
approximates the constraint from the attacker's own clean update, prior local
updates, and prior local scores. Phase 2 (`phase2_cohort_aware`) adds a second
client message: the server relays the current peer updates and prior detector
history to the attacker, which evaluates the exact multi-signal score before
returning a constrained update. This is an intentionally strong white-box
simulation that discloses peer updates to the attacker; it is not a
privacy-preserving or decentralized protocol. They have not been validated in
a confirmatory experiment. The existing one-seed pilot exercised both code
paths, but is only a plumbing check.
Phase 1's surrogate score is not numerically interchangeable with the exact
cohort score, so it needs its own development calibration and must not be
claimed to satisfy the server detector constraint unless server-side logs
confirm acceptance.

Both modes require delta representation and an explicit attack start round.
Phase 2 constrains against the multi-signal score even when evaluated against
another selection/aggregation arm. The local frontend does not expose the new
settings. Do not describe the current candidate, either attack, or old pilot as
empirical evidence.

The code now includes a within-invocation paired-suite runner, cutoff
calibration, run summarization with seed-paired bootstrap intervals, and
Krum, Multi-Krum, coordinate-median, and trimmed-mean update-space baselines.
The suite sequentially resets each arm's initial model and RNG state while
using one Flower Grid and fixed partition assignment. A no-training preflight
probes the node-to-partition mapping; the attacker is anchored to a partition
ID, while maliciousness labels still come from the server-designated Flower
node ID. Each arm's recorded mapping is checked against the preflight mapping.
Completed arms are checkpointed atomically and verified before a resumed suite
skips them.
A one-seed, one-round no-attack smoke suite completed for no filtering,
multi-signal filtering, oracle exclusion, and Krum. A second development
plumbing pilot completed one seed × three rounds × four data regimes × three
attack conditions (none, Phase 1, Phase 2), for 12 runs. All rounds produced
finite aggregates, and both attacks applied their configured optimization
steps. The designated attacker was retained in all arms. The pilot is too
small for scientific comparisons or a paper threshold.

An earlier, superseded attack-informed calibration calculation proposed
8.40045 from the one-seed pilot, with zero observed malicious detections at
that post-hoc cutoff. That value is not a candidate for freezing: it used
adaptive attack scores from the same pilot and is circular. The corrected
calibration uses only benign-only records and has not yet been run on
five-seed development data. No cutoff was frozen and no confirmatory run was
launched. Windows Ray teardown emitted access-violation warnings despite
successful arm completion and artifact writing.

## Resuming suites and Linux execution

The paired runner accepts `suite-resume-manifest` to continue an interrupted
suite. It rejects changed run configuration, source fingerprint, Python /
Flower / PyTorch / NumPy versions, calibration hash or threshold, or partition
IDs. It verifies each manifest-listed run artifact and its expected round
outcomes before skipping that arm. Keep the complete suite directory and its
relative artifact paths on persistent storage. The analysis command rejects
incomplete manifests; partial run records remain available for diagnostics but
must not be presented as complete suite results.

The README contains generic Linux setup, logging, and resume commands. No cloud
resources have been provisioned, no Linux execution has been started, and no
additional training or dataset download was performed for resumability work.
The current runner is sequential and does not configure Ray GPU resources;
Linux placement alone is not evidence of a faster run. Preserve the runtime
versions recorded by the manifest when resuming.

## Stage 1: offline validation

```powershell
python -m unittest discover -s tests -v
```

This checks tensor representation, both adaptive attack constraints,
partitioner construction, robust aggregation baselines, calibration and paired
analysis helpers, survivor weights, oracle identity, and explicit round
outcomes without loading CIFAR-10.

## Stage 2: one-round smoke test

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
flwr run . --stream --run-config "num-server-rounds=1 selection-mode='none' attack-start-round=999 save-model=false"
```

This verifies simulation wiring, centralized evaluation, and per-run artifact
creation. It may download CIFAR-10 on a cache miss. It is not an experiment
result and must not be included in paper tables.

## Stage 3: controlled scientific runs

For each predeclared seed and attack configuration, execute all controls with
the same initial seed, client/data partition setup, number of rounds, attack
timing, and attack scale:

1. No filtering.
2. Cosine detector.
3. Multi-signal detector.
4. Server-side oracle exclusion.
5. Random same-size exclusion.
6. Benign-only aggregation.
7. Krum, Multi-Krum, coordinate median, and trimmed mean.

Keep representation and reference modes fixed within a comparison. Use
`representation-mode='delta'` for the primary update-space study. If the
legacy full-model implementation is reported, run it as a separately labeled
comparison, not as a delta-space replicate. Do not report one smoke run as multi-seed evidence.

The paired runner executes requested arms on one Flower Grid per invocation,
reusing node IDs and partition configuration while explicitly resetting
initial-model and client RNG states for each arm. The attacker node remains
fixed within a suite. The mapping in run artifacts is reported by simulation
clients and checked for consistency; it is not used to determine maliciousness.
Do not treat separate `flwr run` invocations as paired. The earlier five-arm
pilot remains exploratory, matched-config evidence only—not paired causal
evidence. Validate one small paired suite before launching development sweeps.

Oracle controls are simulation-only: ground truth is derived from the
server-designated attacker Flower node ID. Do not add a client-provided
maliciousness metric. Use a separate, predeclared server node ID per run if
attacker identity needs balancing across client nodes.

## Required artifact review

Retain each unique `<run-output-dir>/<run_id>/run.json`. Set
`run-output-dir` to an absolute repository path when launching from Flower's
isolated app environment. Before summarizing results, verify
that comparison records agree on seed, initial-model hash, representation,
reference, attack scale, round count, dataset/partition details, software
versions, client identities, and data assignments, except for the intended
selection-mode difference. Matching only seed and initial-model hash is not
enough. Report each round's explicit status and review per-client
scores/selections, normalized aggregation weights, model and update
diagnostics, non-finite tensor names/locations, and centralized task
accuracy/loss.

The current code records unset values for provenance that the project does not
pin, including the dataset revision and code revision when no Git metadata is
available. Resolve these provenance gaps before claiming full reproducibility.

## Existing pilot: exploratory only

The one-seed, three-round, `gamma=-7`, threshold-0.7 pilot produced run records
under `results/experiment_artifacts/exploratory_delta_pilot/` for all five
selection modes. The initial model hashes and
configured seeds match, but registered node IDs differ, so the runs are not
strictly paired. The detector filtered all clients in rounds 1 and 2 in its
run; random-same-size selection also selected nobody in all three rounds. The
oracle and benign-only controls each retained nine clients per round in their
respective runs. All reported aggregate tensors were finite. These are
descriptive pilot observations from one seed, not paper findings: the client
random streams and attacker-to-data-partition assignment were not held fixed
across the five runs, and there is no multi-seed uncertainty estimate.

Ray emitted Windows access-violation warnings during worker teardown even
though each Flower command exited successfully and wrote its run artifact.
Record this platform behavior and investigate it before relying on larger
Windows-based sweeps.

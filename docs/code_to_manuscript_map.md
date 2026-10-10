# Code-to-manuscript map

This is a review aid, not a manuscript revision. Existing run files and figures
are retained unchanged. Historical results describe the legacy full-model code
path only; they are not evidence for the new delta-space experiment.

## Current implementation definitions

| Manuscript term | `legacy_full_model` | `delta` |
|---|---|---|
| Client payload | Locally trained `state_dict`, optionally multiplied tensor-by-tensor by `attack_scale` | `global_state + attack_scale * (local_state - global_state)` for floating tensors; non-floating buffers are copied from the received global state and not advanced |
| Detector vector | Flattened submitted model tensors | Flattened `local_state - received_global_state` over floating tensors |
| `current_mean` reference | Unweighted mean of valid submitted full-model vectors, including the attacker | Unweighted mean of valid submitted delta vectors, including the attacker |
| Historical reference | `previous_global_model`: last accepted full model; round 1 falls back to current mean | `previous_accepted_delta`: last accepted aggregate displacement; round 1 falls back to current mean |
| Decision | Flag iff cosine score is strictly less than `cosine-threshold` | Same |
| Aggregation | Flower FedAvg over selected model payloads, weighted by `num-examples` and renormalized over survivors | Same payload aggregation; submissions encode attacked client models |
| Oracle identity | Not available from a trusted partition-to-node mapping in this project | Server-designated Flower `node_id`, recorded in run metadata; never inferred from reply metrics |
| Norm diagnostics | Aggregate model-state norm and max absolute parameter | Aggregate model-state diagnostics plus aggregate displacement/update diagnostics |

The Flower strategy skips a round when there are no selected/scoreable clients
or when the aggregate is non-finite. Flower's `Strategy.start` retains its
previous accepted arrays when `aggregate_train` returns `None`; this skip does
not itself stop the whole run. Round outcomes are recorded separately from
Flower's numeric `MetricRecord` using string statuses in the run artifact and a
numeric status code in Flower metrics.

## Follow-up candidate (not paper-validated)

The `multi_signal` selection mode is a new experimental heuristic and is
separate from the cosine detector. It scores flattened floating-point deltas
against the current cohort's geometric median. Its score is the equal-weight
mean of available nonnegative robust-z direction, absolute log-norm, temporal
drift, and client-history deviations; it flags only when that score is strictly
greater than `multi_signal_threshold`. The first round has no temporal or
history signal. The default cutoff is provisional and must be tuned only on
development seeds before confirmatory runs.

The task loader supports seeded Dirichlet label partitions in addition to the
existing IID partitioner. Client RNG streams are now keyed by partition ID
rather than Flower node ID. The selected attacker and oracle label remain
server-designated by Flower node ID, so the partition attacked may still vary
between separate `flwr run` invocations. Per-client records include the
reported partition ID for audit, but that field is not used as a maliciousness
label.

Two score-constrained local-loss attacks are implemented but not yet validated
with Flower training runs. `phase1_own_history` approximates the score
constraint using only the attacker's clean update and local update/score
history. `phase2_cohort_aware` adds a second train message and relays peer
updates plus server-maintained detector history to the attacker, which
constrains against the exact multi-signal scoring function. Phase 2 is an
explicit white-box update-disclosure threat model, not a privacy-preserving FL
protocol. The current code still lacks paired multi-arm execution and
classical robust-aggregation baselines.

## Evidence boundaries and manuscript corrections to consider

| Existing claim/topic | Current evidence status | Required action before publication |
|---|---|---|
| The implementation is FedVault | The repository labels previously used this name, but contains no upstream URL, revision, or source citation establishing identity. The active implementation is now explicitly named `CosineAuditStrategy`. | Call it an audited implementation unless primary-source provenance is supplied. |
| The original deployed defense computed flags but did not enforce them | The current source snapshot had both enforce and non-enforce branches; it does not prove undocumented historical behavior. | Provide the historical source/version or narrow the claim to the inspected snapshot/mode. |
| Similarity is computed on client updates | This was inaccurate for historical runs: the old server flattened full submitted model states. | Label historical data as full-model-state scoring/scaling. Delta-mode claims require new runs. |
| Negative `gamma` attack scales updates | Historical client code multiplied all returned model-state tensors. Delta mode now attacks the displacement relative to the received global state; these are distinct interventions. | Define both separately and do not pool results. |
| “Update norm” | Historical diagnostics used the flattened aggregate model state, not its displacement. | Use “model-state norm” for historical runs; new update norms must identify the exact displacement definition. |
| Non-finite parameters imply training-ending failure | Previous logs show non-finite detector scores and model behavior but do not establish that the run stopped. The new strategy explicitly skips accepting a non-finite aggregate and preserves the previous accepted model; it continues rounds. | Report non-finite state and explicit round status, not “training-ending,” unless run orchestration actually terminates and records that event. |
| Detector selection caused task failure | Old structured logs lack centralized task metrics and tensor-level locations; a finite oracle example does not establish a general causal effect. | New paired runs must include task loss/accuracy and invalid-tensor diagnostics. |
| Paired oracle result | Historical oracle selection read the client-provided `is_malicious` metric, which is not trusted ground truth. | Do not treat historical oracle records as validated; use server-designated node IDs for new IID simulations. |
| Seed counts and reproducibility | Existing run records do not fully identify seeds, data revisions, exact software environment, or pairing. | Treat historical seeds/pairing as unverified unless independent provenance exists. New records state the configured seed and software versions; deterministic execution is not guaranteed. |
| Flower version | Project dependency range and version target differed from the manuscript's claimed exact version. | Cite the run-record version for each new result, not the manuscript's intended version. |
| CIFAR-10 task utility | New run code records server evaluation metrics when a run is actually executed. Unit tests do not load CIFAR-10 and are not task-performance evidence. | Do not add or revise empirical claims until approved runs produce metrics and artifacts. |

No manuscript text has been edited. The current-round delta experiment is a new
estimand selected for the next study; upstream defense identity and literature
novelty remain unverified.

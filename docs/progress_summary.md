# Research Code Progress Summary

**Updated:** 2026-10-07  
**Project:** `cosine-filter-audit`

## Agreed study scope

- Centralized, synchronous FedAvg-style federated learning on CIFAR-10.
- The detector uses client updates (deltas), not full model states.
- “Rootless” means the detector does not use trusted server training data; a
  central coordinator is still present. The centralized CIFAR-10 test set is
  used for task evaluation only.
- Compare an IID control with seeded Dirichlet label-skew partitions at
  α = 1.0, 0.3, and 0.1.
- Start with untargeted poisoning. Backdoor, asynchronous, and decentralized
  settings are later work.
- The follow-up detector is an experimental multi-signal candidate. Its
  threshold is provisional and must be selected on development seeds and
  frozen before confirmatory evaluation.

## Implemented

### Multi-signal update detector

The `multi_signal` mode scores flattened floating-point client deltas using a
current-round geometric-median reference. It combines the available robustly
standardized signals with equal weight:

1. Update distance from the cohort reference.
2. Absolute log update-norm deviation.
3. Temporal drift from the client's prior update.
4. Deviation from the client's historical score.

Temporal and history signals are unavailable in the first round. The default
cutoff (`3.5`) is a placeholder, not an empirically selected value. Detector
selection requires delta representation.

### Adaptive attack phases

Both modes maximize cross-entropy on the malicious client's own local training
data, subject to a score constraint. They require
`selection-mode='multi_signal'`, delta representation, and an explicit attack
start round.

- **Phase 1 — `phase1_own_history`:** approximates the detector constraint from
  the attacker's own clean update and local update/score history. This surrogate
  score is not numerically interchangeable with the cohort detector score.
- **Phase 2 — `phase2_cohort_aware`:** adds a server-to-attacker message with
  current peer updates and detector history, then constrains against the exact
  multi-signal scoring function. This is a deliberately strong white-box
  threat model that reveals peer updates to the attacker; it is not a
  privacy-preserving protocol.

Attack outcomes, scores, local loss before/after optimization, and applied
steps are recorded. Client history is stored locally in Flower's client
context. Attack modes are opt-in; the default remains `none`.

### Data and run reproducibility support

- Added seeded Dirichlet label partitioning alongside IID partitioning.
- Client RNG streams are keyed by partition ID rather than ephemeral Flower
  node ID.
- Server records the observed client partition ID and retains server-side
  attacker labels; client-supplied maliciousness labels are not trusted.
- Added an absolute `run-output-dir` option so Flower runs can save artifacts
  directly to a chosen location. The default path is relative to Flower's
  installed app directory, not necessarily the source checkout.
- Updated protocol and README documentation. The local frontend was not
  modified and does not expose the new settings.

## Paired evaluation and robust baselines

- Added a within-invocation paired-suite runner for no filtering, cosine and
  multi-signal selection, server-side oracle exclusion, random same-size
  exclusion, benign-only aggregation, and Krum, Multi-Krum, coordinate median,
  and trimmed mean.
- Suite arms run sequentially on one Flower Grid with a fixed attacker node,
  seeded initial model, reset RNG streams, and recorded client-to-partition
  mapping. The mapping is checked for stability but is not used as an oracle
  label; attacker labels come from the server-designated node ID.
- Added development-only benign-score threshold calibration. It selects the
  least restrictive cutoff meeting the agreed 5% benign false-rejection
  ceiling. Adaptive-attack scores are excluded because Phase 2 optimizes
  against the configured threshold; using them to select the same threshold
  would be circular. Calibration requires all four regimes and at least five
  development seeds; confirmatory suites reject ineligible calibration and
  overlapping development seeds.
- Added paired run summaries, bootstrap intervals, software/configuration
  provenance, and robust update-space aggregation baselines. These robust
  aggregators are unweighted; FedAvg uses renormalized sample-count weights.
- The adaptive attacks and suite controls are opt-in. Default project
  configuration does not launch a suite or enable an attack.

## Validation completed

- The Flower app packaging build passed.
- All application modules compile with Python.
- **59 offline tests passed.** They cover score behavior, both attack constraints, client
  history, partitioners, robust aggregation, calibration, paired analysis,
  oracle identity, filtering outcomes, and the local UI.
- A one-seed, three-round development plumbing pilot completed across IID and
  Dirichlet α = 1.0, 0.3, and 0.1, with no attack and both adaptive attack
  phases. The 12 runs are under
  `runs/paired_suites/20261007T102954Z_a8573c0d/`; every run has three
  `AGGREGATE_FINITE` round outcomes. Both attack phases applied three
  optimization steps in attacked rounds, and the designated attacker was
  retained in all arms. This is only a workflow/plumbing check, not a
  robustness or non-IID finding.
- An earlier, superseded attack-informed calibration calculation on those
  records proposed `8.40045` with 0% benign false rejection and 0% attacker
  detection. It used the same pilot's adaptive attack scores and is not
  scientifically suitable for freezing a cutoff. The corrected benign-only
  calibration has not yet been run on the required five-seed development set.
  **No cutoff was frozen and no confirmatory runs were launched.**
- A one-seed, one-round Flower paired-suite smoke run completed for no
  filtering, multi-signal selection, server-side oracle exclusion, and Krum.
  It wrote four run records and a suite manifest under
  `runs/paired_suites/20261007T094216Z_0ef9c4af/`. All four rounds produced
  finite aggregates. In this no-attack plumbing run, round-one accuracy/loss
  were 18.18%/2.2643 (no filtering and multi-signal), 16.79%/2.2648 (oracle),
  and 15.81%/2.2652 (Krum). The detector flagged no clients; oracle still
  excluded the designated node by construction. These are not attack-defense
  findings or a meaningful estimate of comparative task performance.
- The first suite attempt exposed an uninitialized client history-arm ID when
  attacks were disabled. The default ID is now initialized for all delta runs;
  the successful rerun exercised the fixed path. Ray emitted Windows access
  violation messages during teardown despite completing all four arms and
  writing the manifest.
- One-round Flower smoke runs completed for both attack phases, with a finite
  aggregate and centralized evaluation. The run records are:
  - `runs/20261007T090041Z_ed32f3a6304e/run.json` — Phase 1.
  - `runs/20261007T090413Z_6edd7079a58f/run.json` — Phase 2.
- In both smoke runs the attacker completed three optimization steps and was
  selected by the detector. The Phase 1 server score was about `0.0085`;
  Phase 2's exact cohort score was about `0.3541`. These values are not
  directly comparable because the attack scoring differs.
- Global CIFAR-10 accuracy after one round was about `18.8%` in both runs.
  This is plumbing output only, not evidence of attack efficacy or defense
  quality.

## Resumable Linux execution

- Added per-arm checkpoints in `suite_manifest.json`, written atomically after
  each completed arm. Resuming validates the suite settings, source fingerprint,
  calibration hash and threshold, exact Python/Flower/PyTorch/NumPy versions,
  partition IDs, and each prior run record before skipping it.
- Added a no-training partition-mapping preflight so a restarted Flower Grid
  can rebind the fixed attacker partition to its current node ID without
  downloading CIFAR-10 or training a model. Oracle labels continue to use the
  server-designated node ID, not a client-reported maliciousness value.
- Analysis now rejects incomplete manifests; completed run records remain
  individually inspectable for diagnostics.
- Added generic Linux setup, persistent-storage, logging, and resume commands
  to the README. No cloud resources, dataset downloads, environment changes,
  or training runs were started for this work.
- **Validation after these changes:** 64 offline tests passed; Python module
  compilation passed; `flwr build` succeeded with UTF-8 console encoding. A
  first build invocation wrote the FAB but its success banner failed under the
  Windows legacy console encoding; rerunning with UTF-8 completed successfully.

## Limitations and next research work

- The smoke runs were not paired: the selected attacker was assigned to a
  different partition in each invocation. Do not interpret their difference
  causally or report them as findings.
- The score cutoff has not been calibrated and frozen. A development replay
  at the candidate cutoff is required before confirmatory use.
- The paired suite has only one seed and three rounds per regime; it does not
  provide robust uncertainty estimates. The paired analysis CLI still needs
  evaluation across a proper multi-seed confirmatory suite.
- Adaptive attacks have only this short development plumbing run, not
  multi-seed adversarial-efficacy evaluation.
- Flower/Ray emitted Windows access-violation warnings during worker teardown.
  Both smoke runs nevertheless completed aggregation, evaluation, and artifact
  writing; investigate this platform behavior before larger sweeps.
- Literature references and claims of novelty have not been independently
  verified here. Do not present the candidate as a validated defense or claim
  publication-level findings from these smoke runs.

## Cost and data note

The two smoke runs launched Flower and used CIFAR-10; Flower created isolated
run environments. They were run after explicit approval. No repeated or
multi-seed training sweep has been launched. Further experiment sweeps should
wait until the threshold, stable attacker-to-partition assignment, pairing,
and baseline plan are fixed.

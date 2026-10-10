"""
verify_enforce.py

Historical synthetic-vector probe. It is not an exact reproduction of the
full-model implementation or the current Flower strategy, and its output is
not CIFAR-10 experimental evidence. The vectors below are a simple proxy for
client deltas, used to explore the geometry of a mean reference without
loading a dataset.

Benign client updates are modeled as: a slowly-shrinking shared
direction (representing convergence) + independent per-client Gaussian
noise (representing SGD/mini-batch variance) -- a standard proxy for
"benign clients that are similar but not identical," which is the
property the defense actually depends on, not CIFAR-10 specifically.

CAVEAT: this probe is not an implementation validation and does not verify
results in any manuscript table.
"""
import numpy as np

np.random.seed(0)

K = 10  # clients, matches paper
ATTACKER_ID = 0
GAMMA = -7.0
THRESHOLD = 0.7
DIM = 500  # synthetic "flattened model" dimensionality
N_ROUNDS = 8


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def run(enforce: bool, use_previous_global: bool, seed: int):
    rng = np.random.default_rng(seed)
    # Shared "true" direction all benign clients are converging toward,
    # shrinking each round to mimic convergence (smaller updates late
    # in training) -- same qualitative behavior as real SGD.
    base_direction = rng.normal(size=DIM)
    base_direction /= np.linalg.norm(base_direction)

    previous_global_flat = None
    log = []

    for t in range(1, N_ROUNDS + 1):
        magnitude = 1.0 / t  # updates shrink as training progresses
        updates = {}
        for k in range(K):
            noise = rng.normal(scale=0.15, size=DIM)
            updates[k] = magnitude * base_direction + noise
        # Attacker: sign-flip + scale applied to its synthetic client vector.
        updates[ATTACKER_ID] = GAMMA * updates[ATTACKER_ID]

        # --- reference point selection: verbatim from custom_strategy.py ---
        if use_previous_global and previous_global_flat is not None:
            reference_vector = previous_global_flat
        else:
            reference_vector = np.mean(list(updates.values()), axis=0)

        # --- detection: verbatim ---
        flagged = set()
        sims = {}
        for k, vec in updates.items():
            sim = cosine_similarity(vec, reference_vector)
            sims[k] = sim
            if sim < THRESHOLD:
                flagged.add(k)

        # --- enforcement: verbatim branching from aggregate_train ---
        if enforce:
            survivors = [k for k in updates if k not in flagged]
            if not survivors:
                log.append((t, sims, flagged, "ALL_FLAGGED_SKIP"))
                # global model does not advance; previous_global_flat unchanged
                continue
        else:
            survivors = list(updates.keys())  # bug reproduction

        aggregated = np.mean([updates[k] for k in survivors], axis=0)
        previous_global_flat = aggregated  # updates the anchor for next round

        log.append((t, sims, flagged, "ok"))

    return log


def summarize(label, log):
    print(f"\n=== {label} ===")
    print(f"{'Round':>5} | {'Attacker sim':>12} | {'Attacker flagged?':>18} | Benign flagged (of 9)")
    for t, sims, flagged, status in log:
        atk_sim = sims[ATTACKER_ID]
        atk_flagged = ATTACKER_ID in flagged
        benign_flagged = len([k for k in flagged if k != ATTACKER_ID])
        print(f"{t:>5} | {atk_sim:>12.4f} | {str(atk_flagged):>18} | {benign_flagged:>2} {'(' + status + ')' if status != 'ok' else ''}")


if __name__ == "__main__":
    for enforce in (True, False):
        for use_prev in (False, True):
            ref_label = "r_prev" if use_prev else "r_mean"
            label = f"enforce={enforce}, reference={ref_label}"
            log = run(enforce=enforce, use_previous_global=use_prev, seed=42)
            summarize(label, log)
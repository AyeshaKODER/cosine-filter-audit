import unittest
import json
import tempfile
from pathlib import Path

from cosine_filter_audit.calibration import (
    calibrate_run_records,
    select_benign_score_cutoff,
    select_score_cutoff,
)


class CalibrationTests(unittest.TestCase):
    def test_selects_cutoff_with_best_malicious_detection_under_fpr_limit(self):
        result = select_score_cutoff(
            [0.1] * 19 + [1.0],
            [0.8, 0.9, 1.1, 1.2],
            max_benign_false_rejection=0.05,
        )

        self.assertEqual(result["threshold"], 0.1)
        self.assertEqual(result["development_benign_false_rejection_rate"], 0.05)
        self.assertEqual(result["development_malicious_detection_rate"], 1.0)

    def test_rejects_empty_or_nonfinite_calibration_data(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            select_score_cutoff([], [1.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            select_score_cutoff([float("nan")], [1.0])

    def test_benign_cutoff_meets_fpr_limit_without_attack_scores(self):
        result = select_benign_score_cutoff(
            [float(value) for value in range(1, 21)],
            max_benign_false_rejection=0.05,
        )

        self.assertEqual(result["threshold"], 19.0)
        self.assertEqual(result["development_benign_false_rejection_rate"], 0.05)

    def test_benign_cutoff_rejects_empty_or_nonfinite_scores(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            select_benign_score_cutoff([])
        with self.assertRaisesRegex(ValueError, "finite"):
            select_benign_score_cutoff([float("inf")])

    def test_calibration_uses_benign_only_records_and_requires_full_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for seed in range(5):
                for regime in ("iid", "1.0", "0.3", "0.1"):
                    record = {
                        "suite_id": "dev-123",
                        "config": {
                            "suite_phase": "development",
                            "attack_mode": "none",
                            "selection_mode": "multi_signal",
                            "seed": seed,
                            "partitioning_mode": (
                                "iid" if regime == "iid" else "dirichlet"
                            ),
                            "dirichlet_alpha": (
                                None if regime == "iid" else float(regime)
                            ),
                        },
                        "rounds": {
                            "1": {
                                "clients": {
                                    "1": {"is_attacker": True, "score": 1.0},
                                    "2": {"is_attacker": False, "score": 0.1},
                                }
                            }
                        },
                    }
                    path = Path(folder) / f"{seed}-{regime}.json"
                    path.write_text(json.dumps(record), encoding="utf-8")
                    paths.append(path)

            attack_record = {
                "suite_id": "dev-123",
                "config": {
                    "suite_phase": "development",
                    "attack_mode": "phase2_cohort_aware",
                    "selection_mode": "multi_signal",
                    "seed": 0,
                    "partitioning_mode": "iid",
                },
                "rounds": {
                    "1": {"clients": {"1": {"is_attacker": True, "score": 1000.0}}}
                },
            }
            attack_path = Path(folder) / "attack.json"
            attack_path.write_text(json.dumps(attack_record), encoding="utf-8")

            result = calibrate_run_records([*paths, attack_path])

        self.assertEqual(result["development_suite_id"], "dev-123")
        self.assertEqual(result["benign_score_count"], 40)
        self.assertEqual(result["threshold"], 1.0)
        self.assertEqual(result["calibration_status"], "ready_for_confirmatory")
        self.assertEqual(result["development_seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(len(result["development_regimes"]), 4)


if __name__ == "__main__":
    unittest.main()

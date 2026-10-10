import json
import tempfile
import unittest
from pathlib import Path

from cosine_filter_audit.analysis import summarize_suite


class AnalysisTests(unittest.TestCase):
    def test_rejects_incomplete_suite_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = Path(folder) / "suite_manifest.json"
            manifest_path.write_text(
                json.dumps({"suite_status": "in_progress", "run_files": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "incomplete suite"):
                summarize_suite(manifest_path)

    def test_summarizes_runs_and_paired_seed_differences(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {
                "suite_id": "suite-1",
                "suite_phase": "confirmatory",
                "pairing_scope": "same grid",
                "seeds": [1, 2],
                "regimes": ["iid"],
                "run_files": [],
            }
            for seed, cosine_acc, multi_acc in [(1, 0.5, 0.6), (2, 0.7, 0.8)]:
                for arm, accuracy, benign_selected in [
                    ("cosine", cosine_acc, False),
                    ("multi_signal", multi_acc, True),
                ]:
                    run_id = f"seed{seed}_iid_none__{arm}"
                    run_path = root / f"{run_id}.json"
                    record = {
                        "suite_id": "suite-1",
                        "config": {
                            "seed": seed,
                            "num-server-rounds": 1,
                            "selection_mode": arm,
                            "aggregation_mode": "fedavg",
                        },
                        "rounds": {
                            "1": {
                                "round_status": "AGGREGATE_FINITE",
                                "clients": {
                                    "1": {
                                        "is_attacker": False,
                                        "score": 0.2,
                                        "selected": benign_selected,
                                    },
                                    "2": {
                                        "is_attacker": True,
                                        "score": 0.4,
                                        "selected": True,
                                    },
                                },
                            }
                        },
                        "server_evaluation_metrics": {
                            "1": {"accuracy": accuracy, "loss": 1.0}
                        },
                    }
                    run_path.write_text(json.dumps(record), encoding="utf-8")
                    manifest["run_files"].append(
                        {
                            "seed": seed,
                            "regime": "iid",
                            "arm_id": f"none__{arm}",
                            "path": str(run_path),
                        }
                    )
            manifest_path = root / "suite_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = summarize_suite(manifest_path)

        self.assertEqual(result["run_count"], 4)
        effect = result["paired_effects"][
            "iid/none/multi_signal-minus-cosine/final_accuracy"
        ]
        self.assertEqual(effect["paired_seed_count"], 2)
        self.assertAlmostEqual(effect["mean_difference"], 0.1)


if __name__ == "__main__":
    unittest.main()
